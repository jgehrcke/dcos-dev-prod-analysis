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
import itertools
import os
import re
import subprocess
import sys
import textwrap
import time
import multiprocessing
import pickle

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


logfmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
datefmt = "%y%m%d-%H:%M:%S"
logging.basicConfig(format=logfmt, datefmt=datefmt, level=logging.INFO)
log = logging.getLogger()


def main():

    prs_downstream = load_prs_from_file(
        'dcos-enterprise_pull-requests-with-comments.pickle')

    prs_upstream = load_prs_from_file(
        'dcos_pull-requests-with-comments.pickle')

    # Perform override command analysis for all pull requests.
    prs_for_comment_analysis = [
        pr for pr in
        itertools.chain(prs_downstream.values(), prs_upstream.values())
    ]
    analyze_pr_comments(prs_for_comment_analysis)

    # analyze_merged_prs(allprs)


def analyze_pr_comments(prs):
    """
    Analyze the issue comments in all pull request objects in `prs`.

    Args:
        prs: a `list` of `github.PullRequest.PullRequest` objects, whereas
            each of these is expected to be enriched with the non-canonical
            `_issue_comments` attribute, expected to be a list of objects of
            type `github.IssueComment.IssueComment`, which are all the normal
            comments on the pull request (e.g. review comments are not
            included).
    """

    log.info('Perform comment analysis for %s PRs', len(prs))

    all_pr_comments, all_override_comments = identify_override_comments(prs)

    # (not so) useful all-time stats. Well, useful for sanity-checking the data.
    log.info('Build histograms for all override comments, for sanity check')
    print('\n** Histogram over number of issue comments per pull request:')
    counter = Counter([len(pr._issue_comments) for pr in prs])
    for item, count in counter.most_common(10):
        print('{:>8} PR(s) have {:>3} comment(s)'.format(count, item))

    print('\n** Histogram over JIRA ticket referred to in all override '
          'comments, all data for sanity check')
    print(Counter([oc['ticket'] for oc in all_override_comments]))

    print('\n** Histogram over CI check name referred to in all override '
        'comments, all data for sanity check')
    print(Counter([oc['checkname'] for oc in all_override_comments]))

    print('\n\n\n* Override comment analysis (all-time stats)')
    build_histograms_from_ocs_last_n_days(all_override_comments, 9999)
    build_histograms_from_ocs_in_recent_prs(prs, 9999)

    # Stats extracted from a more narrow time window from the recent past, more
    # relevant in practice.

    print('\n\n\n* Override comment analysis (last 30 days)')
    build_histograms_from_ocs_last_n_days(all_override_comments, 30)
    build_histograms_from_ocs_in_recent_prs(prs, 30)

    print('\n\n\n* Override comment analysis (last 10 days)')
    build_histograms_from_ocs_last_n_days(all_override_comments, 10)
    build_histograms_from_ocs_in_recent_prs(prs, 10)

    #build_histograms_from_ocs_in_recent_prs(prs, 30)
    #build_histograms_from_ocs_in_recent_prs(prs, 10)
    #build_histograms_from_ocs_last_n_days(all_override_comments, 30)
    #build_histograms_from_ocs_last_n_days(all_override_comments, 10)


    # Find first occurrence of individual override tickets, and show the ones
    # that were used for the first time within the last N days.

    collector = defaultdict(list)
    for comment in all_override_comments:
        collector[comment['ticket']].append(comment['comment_obj'].created_at)

    for ticket, created_dates in collector.items():
        earliest_date = min(created_dates)
        log.info('Earliest appearance of ticket %s: %s', ticket, earliest_date)


def build_histograms_from_ocs_in_recent_prs(prs, max_age_days):
    """
    Find pull requests not older than `max_age_days` and extract all override
    commands issued in them. Perform a statistical analysis on this set of
    override commands.
    """
    print(f'** Histograms from override comments in PRs younger than {max_age_days} days')
    now = datetime.now()
    prs_to_analyze = []
    for pr in prs:
        age = now - pr.created_at
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            prs_to_analyze.append(pr)
    build_histograms_from_ocs_in_prs(prs_to_analyze)


def build_histograms_from_ocs_in_prs(prs):
    topn = 10
    print(f'   Top {topn} number of override commands issued per pull request:')
    counter = Counter([len(pr._override_comments) for pr in prs])
    for item, count in counter.most_common(topn):
        print('{:>8} PR(s) have {:>3} override comment(s)'.format(count, item))


def build_histograms_from_ocs_last_n_days(override_comments, n):
    print(f'** Histograms from override comments younger than {n} days')
    now = datetime.now()
    max_age_days = n
    ocs_to_analyze = []
    for oc in override_comments:
        age = now - oc['comment_obj'].created_at
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            ocs_to_analyze.append(oc)
    print(f'** Number of override comments: {len(ocs_to_analyze)}')
    build_histograms_from_ocs(ocs_to_analyze)


def build_histograms_from_ocs(override_comments):
    topn = 10
    print(f'   Top {topn} JIRA tickets used in override comments')
    counter = Counter([oc['ticket'] for oc in override_comments])
    for item, count in counter.most_common(topn):
        print('{:>8} overrides refer to JIRA ticket {:>3}'.format(count, item))

    print(f'   Top {topn} CI check names used in override comments')
    counter = Counter([oc['checkname'] for oc in override_comments])
    for item, count in counter.most_common(topn):
        print('{:>8} overrides refer to CI check {}'.format(count, item))


def identify_override_comments(prs):
    """
    Extract all Mergebot override comments from `prs`.

    Args:
        prs: a `list` of `github.PullRequest.PullRequest` objects, whereas
            each of these is expected to be enriched with the non-canonical
            `_issue_comments` attribute, expected to be a list of objects of
            type `github.IssueComment.IssueComment`, which are all the normal
            comments on the pull request (e.g. review comments are not
            included).

    Returns:
        all_pr_comments, all_override_comments

        Most importantly, however, this function is expected to add the property
        `_override_comments` to each PR object, a (potentially empty) list of
        override comment dictionaries.
    """

    # Create a data structure `all_pr_comments` that is meant to contain all
    # issue comments from all pull requests, in a single list. "Issue comments"
    # on pull request issues do not contain review comments, but just normal
    # comments emitted straight in the main commentary thread.
    all_pr_comments = []
    for pr in prs:
        all_pr_comments.extend(pr._issue_comments)

    # Basic statistics to get a feeling for the data.
    comments_mentioning_an_override = [
        c for c in all_pr_comments
        if '@mesosphere-mergebot override-status' in c.body
    ]

    log.info(
        'Number of comments containing the text '
        '"@mesosphere-mergebot override-status": %s',
        len(comments_mentioning_an_override)
    )

    # Apply heuristics for detecting actual status overrides.
    all_override_comments = []
    for pr in prs:
        pr._override_comments = []
        for comment in pr._issue_comments:
            oc = detect_override_comment(comment, pr)
            if oc is not None:
                all_override_comments.append(oc)
                pr._override_comments.append(oc)

    log.info('Number of override comments: %s', len(all_override_comments))

    # Note(JP): a (desired) side effect here is that `prs` have been modified
    # in-place with the `_override_comments` property. Do not return this to
    # make the side effect more explicit.
    return all_pr_comments, all_override_comments


def detect_override_comment(comment, pr):
    """
    Inspect `comment`, see if it is a Mergebot CI check override comment. Also
    identify override comment attempts (legit attempts to issue an override
    comment), and try to isolate/ignore conversations about override comments
    (where there was no attempt to actually issue an override comment).

    Note: Mergebot itself seems to have the following command detection logic:

        1) The first to tokens of the command are removed:

            return contents.replace(_BOT_NAME, "").replace(
                BotCommand.OVERRIDE_STATUS_KEY, "").strip()

        2) The remainder is called `override_status_value`. The status key and
           the JIRA ticket ID are extracted based on whitespace splitting from
           the right:

            status_key, jira_id = override_status_value.rsplit(maxsplit=1)

    (see https://github.com/mesosphere/mergebot/blob/fbaeecf/mergebot/mergebot/actions/override_status.py#L31)

    That is, if a status key / check name in an issued override command contains
    whitespace it is invalid. In day-to-day business, we seem to have these keys
    with whitespace, and users use them. One goal here is to detect usage of
    invalid status keys / check names (those that contain whitespace).

    Args:
        comment: an object of type `github.IssueComment.IssueComment`
        pr: an object of type `github.PullRequest.PullRequest`

    Returns:
        A dictionary representing the override comment or `None` if no override
        comment was detected.
    """
    # Get comment body (text), strip leading and trailing whitespace before
    # further processing.
    text = comment.body.strip()
    linecount = len(text.splitlines())

    # A checkname is sometimes submitted with whitespace in it, as in this
    # example (taken verbatim from a real-world comment):
    #
    #     @mesosphere-mergebot override-status "teamcity/dcos/test/upgrade/disabled -> permissive" DCOS-17633
    #
    # Not sure if that is valid from Mergebot's point of view, but it is
    # real-world data. Note(JP): the `[A-Za-z]+.*[A-Za-z]+` in the
    # checkname regex group is supposed to make sure that the checkname
    # starts with a word character, ends with a word character, but is
    # otherwise allowed to contain e.g. whitespace characters, even
    # newlines (as of the DOTALL option). Tested this on
    # https://pythex.org/ The following test string:
    #
    # @mesosphere-mergebot override-status Foo1 foo2
    # bar
    # ticket
    #
    # parses checkname to `Foo1 foo2\nbar` and jiraticket to `ticket`.
    #
    # Additionally, also consider checknames which start/end with `"` (these
    # are invalid, and the goal is to detect them).
    regex = (
        '@mesosphere-mergebot(\s+)override-status(\s+)'
        '(?P<checkname>["A-Za-z]+.*["A-Za-z]+)(\s+)(?P<jiraticket>\S+)'
    )

    match = re.search(regex, text, re.DOTALL)

    if match is not None:

        if 'This repo has @mesosphere-mergebot integration' in text:
            # This is Mergebot's help text comment.
            return None

        if not text.startswith('@mesosphere-mergebot'):
            # This is assumed to be a conversation about an override command,
            # such as one user asking another user to submit one.
            # log.info('Ignore conversation about override comment:\n%s\n', text)
            return None

    if match:

        # Remove URL prefix from ticket name (if present).
        prefix = 'https://jira.mesosphere.com/browse/'
        ticket = match.group('jiraticket')
        if ticket.startswith(prefix):
            ticket = ticket[len(prefix):]

        # Sometimes people have gotten the order of checkname and ticket
        # wrong. Checknames usually contain slashes. Ticket names don't
        # (at this point).
        if '/' in ticket:
            return None

        if linecount > 1:
            ...
            # log.info('Mergebot override in multi-line comment:\n%r', text)

        # Create the override comment data structure used elsewhere in the
        # program.
        override_comment = {
            'prnumber': pr.number,
            'checkname': match.group('checkname').strip(),
            'ticket': ticket,
            'comment_obj': comment
        }

        return override_comment
    return None


def analyze_merged_prs(prs):

    log.info('Filter merged pull requests.')
    filtered_prs = [pr for pr in prs if pr.merged_at is not None]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    log.info('Filter pull requests not created by mergebot.')
    filtered_prs =  [pr for pr in filtered_prs if 'mergebot' not in pr.user.login]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    log.info('Filter pull requests not having `Train` in title.')
    filtered_prs =  [pr for pr in filtered_prs if 'train' not in pr.title.lower()]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    # Proceed with analyzing only those pull requests that were not created
    # by mergebot.

    log.info('Build main Dataframe')

    # I think the point in time when a pull request has been merged is the
    # better reference for determining metrics like throughput and latency than
    # the point in time when a pull request has been created.

    df = pd.DataFrame(
        {
            'created_at': [pr.created_at for pr in filtered_prs],
            'openseconds': [
                (pr.merged_at - pr.created_at).total_seconds() for \
                pr in filtered_prs
            ]
        },
        index=[pd.Timestamp(pr.merged_at) for pr in filtered_prs]
        )

    # Sort by time.
    df.sort_index(inplace=True)

    df['opendays'] = df['openseconds'] / 86400

    matplotlib_config()

    latency = plot_latency(df)

    plt.figure()

    throughput = plot_throughput(filtered_prs)

    plt.figure()

    quality = throughput / latency
    df['quality'] = quality

    plot_quality(df)

    plt.show()


# What is good is low time to merge, and many pull requests merged per time.
# So, divide throughput by latency.


def plot_quality(df):
    df['quality'].plot()
    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day] / latency [day]')
    set_title('PR integration quality for PRs in mesosphere/dcos-enterprise')
    #subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #    matcher.subtitle
    #set_subtitle('Raw data')
    plt.tight_layout(rect=(0,0,1,0.95))


def plot_throughput(filtered_prs):

    # Rolling window of one week width. The column does not matter, only
    # evaluate number of events (rows) in the rolling window, and count them.

    df = pd.DataFrame(
        {
            'foo': [1 for pr in filtered_prs]
        },
        index=[pd.Timestamp(pr.merged_at) for pr in filtered_prs]
        )

    # Sort by time (when the PRs have been merged).
    df.sort_index(inplace=True)

    rollingwindow = df['foo'].rolling('21d')
    throughput = rollingwindow.count()/21.0
    #stddev = rollingwindow.std()

    throughput.plot(
        linestyle='dashdot',
        #linestyle='None',
        #marker='.',
        color='black',
        markersize=5,
    )
    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day]')
    set_title('Pull request throughput for PRs in mesosphere/dcos-enterprise')
    #subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #    matcher.subtitle
    #set_subtitle('Raw data')
    plt.tight_layout(rect=(0,0,1,0.95))

    return throughput


def plot_latency(df):

    df['opendays'].plot(
        #linestyle='dashdot',
        linestyle='None',
        color='gray',
        marker='.',
        markersize=3,
        markeredgecolor='gray'
    )
    plt.xlabel('Pull request creation time')
    plt.ylabel('Time-to-merge latency [day]')
    set_title('Time-to-merge for PRs in mesosphere/dcos-enterprise')
    #subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #    matcher.subtitle
    set_subtitle('Raw data')
    plt.tight_layout(rect=(0,0,1,0.95))

    rollingwindow = df['opendays'].rolling('21d')
    mean = rollingwindow.mean()

    mean.plot(
        linestyle='solid',
        #linestyle='None',
        color='black',
        #marker='.',
        #markersize=1,
        #markeredgecolor='gray'
    )

    #stddev = rollingwindow.std()


    #plt.figure()

    # Rolling window of one week width,

    #rollingwindow = df['opendays'].rolling('7d')
    #mean = rollingwindow.mean()
    #stddev = rollingwindow.std()

    #mean.plot()

    return mean


def set_title(text):
    fig = plt.gcf()
    fig.text(
        0.5, 0.98,
        text,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=14
    )


def set_subtitle(text):
    fig = plt.gcf()
    fig.text(
        0.5, 0.95,
        text,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=10,
        color='gray'
    )


def matplotlib_config():
    matplotlib.rcParams['figure.figsize'] = [10.5, 7.0]
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 150
    #mpl.rcParams['font.size'] = 12

    plt.style.use('ggplot')


def load_prs_from_file(filepath):
    log.info('Unpickle from file: %s', filepath)
    with open(filepath, 'rb') as f:
        return pickle.loads(f.read())


if __name__ == "__main__":
    main()
