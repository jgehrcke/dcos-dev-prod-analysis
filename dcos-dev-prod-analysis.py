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


def load_prs_from_file(filepath):
    log.info('Unpickle from file: %s', filepath)
    with open(filepath, 'rb') as f:
        return pickle.loads(f.read())


def main():

    prs_downstream = load_prs_from_file(
        'dcos-enterprise_pull-requests-with-comments.pickle')

    prs_upstream = load_prs_from_file(
        'dcos_pull-requests-with-comments.pickle')


    # Perform override command analysis for all pull requests (this goes back
    # into the past for quite some time).
    prs_for_override_cmd_analysis = [
        pr for pr in
        itertools.chain(prs_downstream.values(), prs_upstream.values())
    ]
    analyze_mergebot_override_commands(prs_for_override_cmd_analysis)

    # Perform override command analysis for all pull requests created in the
    # last 30 days.

    # analyze_merged_prs(allprs)


def analyze_mergebot_override_commands(prs):

    log.info('Perform Mergebot override command analysis for %s PRs', len(prs))

    all_pr_comments, override_comments = identify_mergebot_override_comments(prs)

    # All-time stats.
    log.info('Histogram over number of issue comments per pull request')
    counter = Counter([len(pr._issue_comments) for pr in prs])
    for item, count in counter.most_common(10):
        print('{:>8} PR(s) have {:>3} comment(s)'.format(count, item))

    log.info(
        'Histogram over number of override commands issued per individual '
        'pull request'
    )
    counter = Counter([len(pr._override_comments) for pr in prs])
    for item, count in counter.most_common(10):
        print('{:>8} PR(s) have {:>3} override comment(s)'.format(count, item))
    log.info('Build histograms for all override comments')
    build_histograms_from_override_comments(override_comments)

    # Stats extracted from a more narrow time window from the recent past, more
    # relevant in practice.

    build_histograms_from_override_comments_last_n_days(override_comments, 30)
    build_histograms_from_override_comments_last_n_days(override_comments, 10)

    # Find first occurrence of individual override tickets, and show the ones
    # that were used for the first time within the last N days.

    collector = defaultdict(list)
    for comment in override_comments:
        collector[comment['ticket']].append(comment['comment_obj'].created_at)

    for ticket, created_dates in collector.items():
        earliest_date = min(created_dates)
        log.info('Earliest appearance of ticket %s: %s', ticket, earliest_date)



def build_histograms_from_override_comments_last_n_days(override_comments, n):
    log.info('Build histograms for override comments younger than %s days', n)
    now = datetime.now()
    max_age_days = n
    override_comments_to_analyze = []
    for oc in override_comments:
        age = now - oc['comment_obj'].created_at
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            override_comments_to_analyze.append(oc)
    build_histograms_from_override_comments(override_comments_to_analyze)


def build_histograms_from_override_comments(override_comments):

    log.info('Histogram over JIRA ticket referred to in the override comments')
    counter = Counter([oc['ticket'] for oc in override_comments])
    print(Counter)
    for item, count in counter.most_common(10):
        print('{:>8} override comments refer to ticket {:>3}'.format(count, item))


    log.info('Histogram over CI check name referred to in the override comments')
    counter = Counter([oc['checkname'] for oc in override_comments])
    print(counter)
    for item, count in counter.most_common(10):
        print('{:>8} override comments refer to CI check {}'.format(count, item))


def identify_mergebot_override_comments(prs):

    # Create a data structure `all_pr_comments` that is meant to contain all
    # issue comments from all pull requests, in a single list. "Issue comments"
    # on pull request issues do not contain review comments, but just normal
    # comments emitted straight in the main commentary thread.
    all_pr_comments = []
    for pr in prs:
        all_pr_comments.extend(pr._issue_comments)

    comments_mentioning_an_override = [
        c for c in all_pr_comments \
        if '@mesosphere-mergebot override-status' in c.body
    ]

    log.info(
        'Number of comments containing the text '
        '"@mesosphere-mergebot override-status": %s',
        len(comments_mentioning_an_override)
    )

    # Analyze PR comments, look for status overrides.
    override_comments = []
    convs_about_override_comments = 0
    for pr in prs:

        # Dynamically load the override comments onto each PR object.
        pr._override_comments = []

        for comment in pr._issue_comments:
            # Strip leading and trailing whitespace.
            text = comment.body.strip()
            linecount = len(text.splitlines())


            # @mesosphere-mergebot override-status
            # "teamcity/dcos/test/upgrade/disabled -> permissive" DCOS-17633
            # (not sure if that is valid from Mergebot's point of view, but it
            # is real-world data)
            regex = '@mesosphere-mergebot(\s+)override-status(\s+)(?P<checkname>.+)(\s+)(?P<jiraticket>\S+)'
            match = re.search(regex, text)


            if match is not None:

                if 'This repo has @mesosphere-mergebot integration' in text:
                    # This is Mergebot's help text comment.
                    continue

                if not text.startswith('@mesosphere-mergebot'):
                    # This is not an actual override command but just a
                    # conversation about one.
                    log.info('Ignore override mention:\nXXXX\n%s\n', text)
                    convs_about_override_comments += 1
                    continue

                if linecount > 1:
                    ...
                    #log.info('Mergebot override in multi-line comment:\n%r', text)
                    #log.info(pr)

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
                    continue

                if '>' in ticket:
                    print('YYYYY')
                    log.info(text)
                    log.info('ticket: `%r`', ticket)
                    log.info('checkname: `%r`', match.group('checkname'))

                override_comment = {
                    'prnumber': pr.number,
                    'checkname': match.group('checkname').strip(),
                    'ticket': ticket,
                    'comment_obj': comment
                }

                override_comments.append(override_comment)
                pr._override_comments.append(override_comment)

    log.info('Number of override comments: %s', len(override_comments))
    log.info(
        'Identified %s mentions as conversations about override comments and '
        'ignored them', convs_about_override_comments
    )

    # Note: a (desired) side effect here is that `prs` have been modified
    # in-place with the `_override_comments` property. Do not return this to
    # make the side effect more explicit.
    return all_pr_comments, override_comments


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


if __name__ == "__main__":
    main()
