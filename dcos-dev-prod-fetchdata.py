#!/usr/bin/env python
# Copyright 2018 Jan-Philip Gehrcke
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import logging
import json
import os
import re
import pickle
import sys
import concurrent.futures

from collections import Counter
from datetime import datetime

from github import Github


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(threadName)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S"
    )


GHUB = Github(
    os.environ['GITHUB_USERNAME'],
    os.environ['GITHUB_APITOKEN'],
    per_page=100
)


def main():
    repo = GHUB.get_organization('mesosphere').get_repo('dcos-enterprise')
    log_remaining_requests()
    fetch_prs_with_comments_for_repo(repo, 'dcos-enterprise')
    log_remaining_requests()

    repo = GHUB.get_organization('dcos').get_repo('dcos')
    log_remaining_requests()
    fetch_prs_with_comments_for_repo(repo, 'dcos')
    log_remaining_requests()


def fetch_prs_with_comments_for_repo(repo, reponame):
    prs = fetch_pull_requests(repo, reponame)
    prs = fetch_comments_for_all_prs(prs, reponame)
    return prs


def fetch_comments_for_all_prs(prs_current_without_comments, reponame):
    # Expect `prs` to be a dictionary with the values being PullRequest objects.

    log.info('Collect issue comments for each PR individually.')

    # name_prefix =
    # today = datetime.now().strftime('%Y-%m-%d')
    # filepath = today + '-' + name_prefix + '.pickle'
    filepath = reponame + '_pull-requests-with-comments.pickle'

    prs_old_with_comments = load_file_if_exists(filepath)

    if prs_old_with_comments is not None:

        # Fetch comments only for new pull requests.
        # `prs_current_without_comments` might be fresher (contain more/newer
        # PRs). `prs_current_without_comments` is a dictionary with the keys
        # being the PR numbers.

        log.info(
            'Loaded %s PRs with comments from disk',
            len(prs_old_with_comments)
        )

        new_pr_numbers = set(prs_current_without_comments.keys()) - \
            set(prs_old_with_comments.keys())

        prs_to_fetch_comments_for = {
            n:prs_current_without_comments[n] for n in new_pr_numbers
        }

        if not prs_to_fetch_comments_for:
            log.info('Nothing to fetch, data on disk is up-to-date')
            return prs_old_with_comments

        log.info(
            'Fetching comments for %s new PRs',
            len(prs_to_fetch_comments_for)
        )

    else:
        prs_to_fetch_comments_for = prs_current_without_comments
        # Fetch comments for all pull requests.

        log.info(
            'Fetching comments for %s PRs',
            len(prs_to_fetch_comments_for)
        )

    fetch_pr_comments_in_threadpool(prs_to_fetch_comments_for)

    if prs_old_with_comments is not None:
        # Combine data loaded from disk, and fresh data from GitHub.
        prs_old_with_comments.update(prs_to_fetch_comments_for)
        prs_with_comments = prs_old_with_comments
    else:
        # There was no old data; `prs_to_fetch_comments_for` is sufficient.
        prs_with_comments = prs_to_fetch_comments_for

    persist_data(prs_with_comments, filepath)
    # log.info('fetch_comments_for_all_prs(): return %s', prs_with_comments)
    return prs_with_comments


def fetch_pr_comments_in_threadpool(prs_to_fetch_comments_for):
    """
    Modify `prs_to_fetch_comments_for` in-place.
    """

    # https://github.com/webuildsg/webuild/issues/290 Github does not allow too
    # many requests being issued concurrently. "We can't give you an exact
    # number here since the limits are not that simple and we'll be tweaking
    # them over time, but I'd recommend reducing concurrency as much as
    # possible, e.g. so that you don't make over ~20 requests concurrently, and
    # then reducing further if you notice that you're still hitting these
    # limits." Note(JP): I tried with 20 and immediately ran into an 'abuse'
    # error message. I tried with 10 and it took 2 minutes to generate an abuse
    # error message. I will retry with 5 and see.
    reqlimit_before = GHUB.rate_limiting[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        # Submit work, build up mapping between futures and pull requests.
        futures_to_prs = {
            executor.submit(fetch_comments_for_pr, pr): pr for _, pr in prs_to_fetch_comments_for.items()
            }

        for future in concurrent.futures.as_completed(futures_to_prs):
            pr = futures_to_prs[future]
            try:
                comments = future.result()
                log.info('Fetched %s comments for pr %s', len(comments), pr)
            except Exception as exc:
                log.error('%r generated an exception: %s' % (pr, exc))

    reqlimit_after = GHUB.rate_limiting[0]
    reqs_performed = reqlimit_before - reqlimit_after
    log.info('Number of requests performed: %s', reqs_performed)



def fetch_comments_for_pr(pr):
    # Executed in a thread as part of a thread pool. Modify object `pr`
    # in-place.
    log_remaining_requests()
    log.info('Fetch comments for pull request %s', pr)
    pr._issue_comments = []
    for comment in pr.get_issue_comments():
        pr._issue_comments.append(comment)

    return pr._issue_comments


def fetch_pull_requests(repo, reponame):

    log.info('Fetch pull requests')
    #today = datetime.now().strftime('%Y-%m-%d')
    #filepath = today + '-' + name_prefix + '.pickle'
    filepath = reponame + '_pull-requests.pickle'

    prs = load_file_if_exists(filepath)
    if prs is not None:
        return prs

    # The following loop emits 10 HTTP requests for fetching data about
    # 1000 pull requests.
    reqlimit_before = GHUB.rate_limiting[0]

    prs = {}
    for count, pr in enumerate(repo.get_pulls('all'), 1):
        # Store `PullRequest` object with integer key in dictionary.
        prs[pr.number] = pr
        if count % 100 == 0:
            log.info('%s pull requests fetched', count)
            # log.info('GH rate limit: %r', GHUB.rate_limiting)

    reqlimit_after = GHUB.rate_limiting[0]
    reqs_performed = reqlimit_before - reqlimit_after
    log.info('Number of requests performed: %s', reqs_performed)

    persist_data(prs, filepath)
    return prs


def log_remaining_requests():
    remaining_requests = GHUB.get_rate_limit().rate.remaining
    log.info('GitHub rate limit remaining requests: %r', remaining_requests)


def load_file_if_exists(filepath):
    # Load from disk if already fetched today, otherwise return `None`.
    if os.path.exists(filepath):
        log.info('Loading data from file: %s', filepath)
        with open(filepath, 'rb') as f:
            data = f.read()
        return pickle.loads(data)
    return None


def persist_data(obj, filepath):
    data = pickle.dumps(obj)
    log.info('Persist %s byte(s) to file %s', len(data), filepath)
    with open(filepath, 'wb') as f:
        f.write(data)


def fetch_issue_comments_and_associate_with_prs_legacy():
    issuecomments = fetch_issue_comments_legacy(repo)

    # Associate comments with pull requests: each "issue comment" listed for the
    # repository contains an `html_url` attribute which refers to the issue/PR
    # number that the comment was posted in
    # Ref: https://developer.github.com/v3/issues/comments/#list-comments-in-a-repository
    # Note(JP): after some investigation I found that this part of the API
    # only reveals those comments of pull requests that are made against the
    # master branch. Comments of pull requests against non-master branches
    # (such as `1.11`) are simply not returned.
    for _, pr in prs.items():
        pr._issuecomments =[]
        # print(pr.comments)

    for ic in issuecomments:
        match = re.search('pull/(?P<prnumber>[0-9]+)', ic.html_url)
        prnumber = int(match.group('prnumber'))
        pr = prs[prnumber]
        pr._issuecomments.append(ic)


def fetch_issue_comments_legacy(repo):

    log.info('Fetch issue comments')
    name_prefix = 'issue-comments-raw'
    today = datetime.now().strftime('%Y-%m-%d')
    filepath = today + '-' + name_prefix + '.pickle'

    issuecomments = load_file_if_exists(filepath)
    if issuecomments is not None:
        return issuecomments

    # The following loop emits 10 HTTP requests for fetching data about
    # 1000 pull request / issue comments.
    reqlimit_before = GHUB.rate_limiting[0]
    issuecomments = []
    for count, comment in enumerate(repo.get_issues_comments(), 1):
        issuecomments.append(comment)
        if count % 100 == 0:
            log.info('%s comments fetched', count)
            log.info('GH rate limit: %r', GHUB.rate_limiting)
    reqlimit_after = GHUB.rate_limiting[0]
    reqs_performed = reqlimit_before - reqlimit_after
    log.info('Number of requests performed: %s', reqs_performed)

    persist_data(issuecomments, filepath)
    return issuecomments


if __name__ == "__main__":
    main()