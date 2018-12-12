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

import argparse
import logging
import os
import pickle
import concurrent.futures
from datetime import datetime

from github import Github
import requests
import retrying


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(threadName)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S"
    )


NOW = datetime.utcnow()
GHUB = Github(
    os.environ['GITHUB_USERNAME'],
    os.environ['GITHUB_APITOKEN'],
    per_page=100
)


def main():

    parser = argparse.ArgumentParser(
        description='Fetch data from a GitHub repository. Requires the '
        'environment variables GITHUB_USERNAME and GITHUB_APITOKEN to be set.')
    parser.add_argument(
        'repo',
        metavar='REPOSITORY',
        help="Organization and repository. Must contain a slash. "
        "Example: coke/truck"
        )
    args = parser.parse_args()

    org, reponame = args.repo.split('/')

    repo = GHUB.get_organization(org).get_repo(reponame)
    log.info('Working with repository `%s`', repo)

    log.info('Request quota limit: %s', GHUB.get_rate_limit())
    # check_request_quota_and_wait()
    fetch_prs_with_comments_for_repo(repo, reponame)


def fetch_prs_with_comments_for_repo(repo, reponame):
    prs = fetch_pull_requests(repo, reponame)
    prs = fetch_details_for_all_prs(prs, reponame)
    return prs


def fetch_details_for_all_prs(prs_current_without_details, reponame):
    # Expect `prs` to be a dictionary with the values being PullRequest objects.

    log.info('Collect issue comments & events for each PR individually.')

    # name_prefix =
    # today = datetime.now().strftime('%Y-%m-%d')
    # filepath = today + '-' + name_prefix + '.pickle'
    filepath = reponame + '_pull-requests-with-comments-events.pickle'

    prs_old_with_comments = load_file_if_exists(filepath)

    if prs_old_with_comments is not None:

        # Fetch comments & events only for new pull requests.
        # `prs_current_without_details` might be fresher (contain more/newer
        # PRs). `prs_current_without_details` is a dictionary with the keys
        # being the PR numbers.

        log.info(
            'Loaded %s PRs with comments & events from disk',
            len(prs_old_with_comments)
        )

        prs_to_fetch_details_for = {}

        # See which PRs are new, compared to what was persisted to disk.
        new_pr_numbers = set(prs_current_without_details.keys()) - \
            set(prs_old_with_comments.keys())

        # For the newly seen PRs we definitely need to fetch comments.
        for n in new_pr_numbers:
            prs_to_fetch_details_for[n] = prs_current_without_details[n]

        log.info(
            'Fetching comments for %s new PRs',
            len(prs_to_fetch_details_for)
        )

        # Note(JP): for the more recent pull requests it is likely that there
        # have been comments incoming after the last update. That is, the fact
        # that a pull request was processed in a previous run does not mean that
        # all of its currently associated issue comments have been fetched. To
        # make this problem go away reliably requires making an HTTP request per
        # pull request which is precisely not what we want to do here (let alone
        # as of GitHub's API rate limit). A 'good enough' best effort approach
        # for now is to look at pull requests from the last 30 days and to fetch
        # all their associated comments. A complete re-fetch of all pull
        # requests and their associated comments every now and then is required
        # to make sure that new comments made in old pull requests are not
        # missed.
        max_age_days = 30
        old_prs_to_analyze = []
        log.info('Identifying most recent PRs')
        for _, pr in prs_old_with_comments.items():
            # `pr.created_at` sadly is a native datetime object. It is known to
            # represent the time in UTC, however. `NOW` also is a datetime
            # object explicitly in UTC.
            age = NOW - pr.created_at
            if age.total_seconds() < 60 * 60 * 24 * max_age_days:
                old_prs_to_analyze.append(pr)

        # If a previous run failed for whichever reason (for example as of the
        # GitHub request quota limit not being properly handled ) then the
        # pickled data might have stored the PR objects, but with the _events
        # and/or the _comments properties missing. For those PRs perform the
        # complete update.
        for _, pr in prs_old_with_comments.items():
            if not hasattr(pr, '_events') or not hasattr(pr, '_comments'):
                # Adding the same PR twice should not be a problem, is
                # uniqueified below.
                old_prs_to_analyze.append(pr)

        log.info('Most recent PRs: %s', old_prs_to_analyze)

        log.info(
            'Fetching comments & events for %s recent PRs',
            len(old_prs_to_analyze)
        )

        for pr in old_prs_to_analyze:
            prs_to_fetch_details_for[pr.number] = pr

        if not prs_to_fetch_details_for:
            log.info('Nothing to fetch, data on disk is up-to-date')
            return prs_old_with_comments

    else:
        prs_to_fetch_details_for = prs_current_without_details
        # Fetch comments for all pull requests.

        log.info(
            'Fetching comments & events for %s PRs',
            len(prs_to_fetch_details_for)
        )

    fetch_pr_details_in_threadpool(prs_to_fetch_details_for)

    if prs_old_with_comments is not None:
        # Combine data loaded from disk, and fresh data from GitHub.
        prs_old_with_comments.update(prs_to_fetch_details_for)
        prs_with_comments = prs_old_with_comments
    else:
        # There was no old data; `prs_to_fetch_details_for` is sufficient.
        prs_with_comments = prs_to_fetch_details_for

    persist_data(prs_with_comments, filepath)
    # log.info('fetch_comments_for_all_prs(): return %s', prs_with_comments)
    return prs_with_comments


def fetch_pr_details_in_threadpool(prs_to_fetch_details_for):
    """
    Modify `prs_to_fetch_details_for` in-place.

    # https://github.com/webuildsg/webuild/issues/290 Github does not allow too
    # many requests being issued concurrently. "We can't give you an exact
    # number here since the limits are not that simple and we'll be tweaking
    # them over time, but I'd recommend reducing concurrency as much as
    # possible, e.g. so that you don't make over ~20 requests concurrently, and
    # then reducing further if you notice that you're still hitting these
    # limits." Note(JP): I tried with 20 and immediately ran into an 'abuse'
    # error message. I tried with 10 and it took 2 minutes to generate an abuse
    # error message. I will retry with 5 and see. Update many weeks later: the
    # current configuration seems to work fine. Update: when fetching only
    # comments for every individual PR the current settings work fine. When
    # additionally also fetching PR events then I ran into
    #
    # 181211-14:05:50.140
    # ERROR:MainThread: PullRequest(title="Refactor dcos_test_utils.marathon",
    # number=1772) generated an exception: 403 {'documentation_url':
    # 'https://developer.github.com/v3/#abuse-rate-limits', 'message': 'You have
    # triggered an abuse detection mechanism. Please wait a few minutes before
    # you try again.'}

    """
    reqlimit_before = GHUB.rate_limiting[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        # Submit work, build up mapping between futures and pull requests.
        futures_to_prs = {
            executor.submit(fetch_details_for_pr, pr): pr
            for _, pr in prs_to_fetch_details_for.items()
            }

        count_total = len(futures_to_prs)
        count_completed = 0
        count_failed = 0

        for future in concurrent.futures.as_completed(futures_to_prs):

            pr = futures_to_prs[future]

            try:
                comments, events = future.result()
                log.info(
                    'Fetched %s comments and %s events for pr %s',
                    len(comments), len(events), pr)

                count_completed += 1

                if count_completed % 20 == 0:
                    log.info('Completed %s of %s PRs', count_completed, count_total)

            except Exception as exc:
                log.error('%r generated an exception: %s' % (pr, exc))
                count_failed += 1

        log.info('Fetching defails succeeded for %s PRs', count_completed)
        log.info('Fetching defails failed for %s PRs', count_failed)

    # Potentially inaccurate log message (if a quota reset happened between
    # `before` and `after`).
    reqlimit_after = GHUB.rate_limiting[0]
    reqs_performed = reqlimit_before - reqlimit_after
    log.info('Number of requests performed: %s', reqs_performed)


def handle_rate_limit_error(exc):

    if 'wait a few minutes before you try again' in str(exc):
        log.warning('GitHub abuse mechanism triggered, wait 60 s, retry')
        return True

    if '403' in str(exc):
        log.warning('Exception contains 403, wait 60 s, retry: %s', str(exc))
        # The request count quota is not necessarily responsible for this
        # exception, but it usually is. Log the expected local time when the
        # new quota arrives.
        unix_timestamp_quota_reset = GHUB.rate_limiting_resettime
        local_time = datetime.fromtimestamp(unix_timestamp_quota_reset)
        log.info(
            'New req count quota at: %s',
            local_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        return True

    # For example, `RemoteDisconnected` is a case I have seen in production.
    if isinstance(exc, requests.exceptions.RequestException):
        log.warning('RequestException, wait 60 s, retry: %s', str(exc))
        return True

    return False


@retrying.retry(
    wait_fixed=60000,
    retry_on_exception=handle_rate_limit_error
)
def fetch_details_for_pr(pr):
    """
    Given a PR explicitly fetch all associated
        - comments
        - events

    Note(JP): GitHub's API for fetching all comments and events for a given
    repository should yield the same (and would save many HTTP requests as of
    better pagnination), but does not yield the same (yields much less data)
    compared to taking this approach here. The difference is as far as I know
    not documented.

    Executed in a thread as part of a thread pool. Modify object `pr`
    in-place.

    This needs more-or-less tight request quota checking.
    """
    log.info('Fetch comments and events for pull request %s', pr)

    # These properties will be serialized upon pickling and can later be
    # inspected in the analysis part of the program.
    pr._issue_comments = []
    pr._events = []

    for comment in pr.get_issue_comments():
        pr._issue_comments.append(comment)

    for event in pr.as_issue().get_events():
        pr._events.append(event)

    return pr._issue_comments, pr._events


def fetch_pull_requests(repo, reponame):

    log.info('Fetch pull requests with pagination (~100 per HTTP request)')

    persist_filepath = reponame + '_pull-requests.pickle'
    prs = load_file_if_exists(persist_filepath)
    if prs is not None:
        # Note(JP): in addition to loading from disk a lightweight best-effort
        # update is performed using the GET /repos/:owner/:repo/pulls call with
        # appropriate usage of the `state` and `sort` parameters. Note however
        # that this is probably still imperfect, with a systematic error
        # accumulating over time. That is, a regular complete refresh of the
        # data is advisable anyway.
        log.info('Get first page of last updated PRs from GitHub')
        lastupdated_prs = repo.get_pulls(
            'all', sort='updated', direction='desc').get_page(0)
        log.info(f'Got {len(lastupdated_prs)} PRs')
        for lastupdated_pr in lastupdated_prs:
            # Replace the PR loaded from disk with its updated variant.
            prs[lastupdated_pr.number] = lastupdated_pr

        log.info('Get first page of last created PRs from GitHub')
        lastcreated_prs = repo.get_pulls('all').get_page(0)
        log.info(f'Got {len(lastcreated_prs)} PRs')
        for lastcreated_pr in lastcreated_prs:
            # Replace the PR loaded from disk with its updated variant.
            prs[lastcreated_pr.number] = lastcreated_pr

        # Return what was read from disk (plus best-effort update)
        return prs

    reqlimit_before = GHUB.rate_limiting[0]
    prs = {}
    for count, pr in enumerate(repo.get_pulls('all'), 1):
        # Store `PullRequest` object with integer key in dictionary.
        prs[pr.number] = pr
        if count % 100 == 0:
            log.info('%s pull requests fetched', count)

    reqlimit_after = GHUB.rate_limiting[0]
    reqs_performed = reqlimit_before - reqlimit_after
    log.info('Number of requests performed: %s', reqs_performed)

    persist_data(prs, persist_filepath)
    return prs


# def check_request_quota_and_wait():
#     """
#     GitHub allows 5000 HTTP requests against its API within 1 hour for 1
#     account. Check quota and proceed if it looks good. Wait for quota to refill
#     if consumed (takes 1 hour at moast).

#     GHUB.get_rate_limit().rate.remaining issues an HTTP request which does not
#     count against the quote. It however consumes time.

#     GHUB.rate_limiting[0] is a local lookup based on internal bookkeeping. Use
#     that so that this function can be called frequently without adding
#     significant run time.
#     """
#     while True:
#         # remaining = GHUB.get_rate_limit().rate.remaining
#         remaining = GHUB.rate_limiting[0]
#         if remaining > 10:
#             log.info('GitHub API req quota: %r, proceed', remaining)
#             break
#         log.info('GitHub API req quota: %r, wait', remaining)
#         time.sleep(60)


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


if __name__ == "__main__":
    main()
