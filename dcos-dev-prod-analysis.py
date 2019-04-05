#!/usr/bin/env python
# Copyright 2018-2019 Jan-Philip Gehrcke
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

"""
Generates a DC/OS developer report.

Input: pull request data from the two DC/OS repositories, as binary data
persisted to disk via pickle.

Output: a Markdown document with tables, and figure files.

The output is meant to be transformed to be into an HTML document via e.g.
pandoc.

Warning: invocation of this program removes the output directory and its
contents.
"""

import argparse
import itertools
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap

from enum import Enum
from io import StringIO
from collections import Counter, defaultdict
from datetime import datetime

import pytablewriter
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S"
    )


NOW = datetime.utcnow()
TODAY = NOW.strftime('%Y-%m-%d')
OUTDIR = None


def main():
    global OUTDIR

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generates a DC/OS developer report as a Markdown document.',
        epilog=textwrap.dedent(__doc__).strip()
    )
    parser.add_argument('--output-directory', default=TODAY + '_report')
    parser.add_argument('--resources-directory', default='resources')
    parser.add_argument('--pandoc-command', default='pandoc')

    args = parser.parse_args()

    prs_downstream = load_prs_from_file(
        'dcos-enterprise_pull-requests-with-comments-events.pickle')

    prs_upstream = load_prs_from_file(
        'dcos_pull-requests-with-comments-events.pickle')

    if os.path.exists(args.output_directory):
        if not os.path.isdir(args.output_directory):
            log.error(
                'The specified output directory path does not point to a directory: %s',
                args.output_directory
            )
            sys.exit(1)

        log.info('Remove output directory: %s', args.output_directory)
        shutil.rmtree(args.output_directory)

    log.info('Create output directory: %s', args.output_directory)
    os.makedirs(args.output_directory)

    OUTDIR = args.output_directory

    # Perform override command analysis for all pull requests.
    prs_for_analysis = [
        pr for pr in
        itertools.chain(prs_downstream.values(), prs_upstream.values())
    ]

    newest_pr_created_at = max(pr.created_at for pr in prs_for_analysis)
    newest_pr_created_at_text = newest_pr_created_at.strftime('%Y-%m-%d %H:%M UTC')
    oldest_pr_created_at = min(pr.created_at for pr in prs_for_analysis)
    oldest_pr_created_at_text = oldest_pr_created_at.strftime('%Y-%m-%d %H:%M UTC')

    now_text = NOW.strftime('%Y-%m-%d %H:%M UTC')
    markdownreport = StringIO()
    markdownreport.write(textwrap.dedent(
    f"""
    % DC/OS developer productivity report
    %
    % {now_text}

    The report is generated based on GitHub pull request data from both DC/OS
    repositories. The code for generating this report lives in
    [`dcos-dev-prod-analysis`](https://github.com/jgehrcke/dcos-dev-prod-analysis).
    The newest pull request considered for this report was created at
    {newest_pr_created_at_text}. The oldest pull request was created at
    {oldest_pr_created_at_text}.


    """
    ).strip())

    # Lazy-perform MPL config (fail fast above).
    matplotlib_config()

    prs_for_throughput_analysis = prs_for_analysis
    prs_for_comment_analysis = prs_for_analysis

    analyze_merged_prs(prs_for_throughput_analysis, markdownreport)
    analyze_pr_comments(prs_for_comment_analysis, markdownreport)

    log.info('Rewrite JIRA ticket IDs in the Markdown report')
    report_md_text = markdownreport.getvalue()
    # Match JIRA ticket string only when there is a leading space
    report_md_text = re.sub(
        " (?P<ticket>[A-Z_]+-[0-9]+)",
        " [\g<ticket>](https://jira.mesosphere.com/browse/\g<ticket>)",
        report_md_text
    )

    md_report_filepath = os.path.join(OUTDIR, TODAY + '_dcos-dev-prod-report.md')

    log.info('Write generated Markdown report to: %s', md_report_filepath)
    with open(md_report_filepath, 'wb') as f:
        f.write(report_md_text.encode('utf-8'))

    log.info('Copy resources directory into output directory')
    shutil.copytree(args.resources_directory, os.path.join(OUTDIR, 'resources'))

    html_report_filepath = os.path.splitext(md_report_filepath)[0] + '.html'
    log.info('Trying to run Pandoc for generating HTML document')
    pandoc_cmd = [
        args.pandoc_command,
        '--toc',
        '--standalone',
        '--template=resources/template.html',
        md_report_filepath,
        '-o',
        html_report_filepath
        ]
    log.info('Running command: %s', ' '.join(pandoc_cmd))
    p = subprocess.run(pandoc_cmd)
    if p.returncode == 0:
        log.info('Pandoc terminated indicating success')
    else:
        log.info('Pandoc terminated indicating error')


def analyze_pr_comments(prs, report):
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

    # All-time stats. These are for example useful for sanity-checking the
    # raw data, and the string parsing heuristics.
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

    print('\n** Histogram over comment creator for all override '
          'comments')
    print(Counter([oc['comment_obj'].user.login for oc in all_override_comments]))

    # Now output the same kinds of stats for different periods of times. Stats
    # extracted from a more narrow time window from the recent past are probably
    # more relevant in practice.

    report.write(textwrap.dedent(
    """

    ## Status check override report (CI instability analysis)

    ### On the significance of override command data

    A status check override command is issued by a human on a pull request via a
    GitHub comment. An override command associates the name of a failed CI check
    (the _check name_) with a JIRA ticket ID. The JIRA ticket tracks a specific
    cause or symptom, usually an instability problem.

    Example: when a developer issues the override command
    ```
    override-status teamcity/dcos/test/dcos-e2e/docker/static/strict https://jira.mesosphere.com/browse/DCOS_OSS-2115

    ```
    they intend to express that

    - the CI check with the check name
      `teamcity/dcos/test/dcos-e2e/docker/static/strict` (which itself is
      comprised of hundreds of individual tests) failed as of a known
      instability in a specific test called `test_vip` (the details can be found
      in the corresponding JIRA ticket DCOS_OSS-2115) and that
    - this failure is unrelated to their patch (which is why they would like to
      _override_ the CI check result from _failed_ to _passed_).

    Each individual override command

    - means that the integration of the pull request it was issued in was
      delayed by one or more CI checks failing as of problems unrelated to the
      patch (that is annoying, isn't it?).
    - was issued by a human after they have analyzed the specific cause or
      symptom of the problem.
    - implies that the corresponding developer spent significant time (minutes
      to hours) figuring out the appropriate override command (this may require
      debugging, doing a JIRA search, creating a JIRA ticket, ...).

    That is, override command data represent real pain experienced by individual
    developers as of CI instabilities. These data are a good source for
    prioritizing work against CI instabilities (which instabilities happen most
    often? which instabilities are new?).

    """
    ))

    # Identify and leave note about newest override command.
    newest_oc_created_at = max(oc['comment_obj'].created_at for oc in all_override_comments)
    oldest_oc_created_at = min(oc['comment_obj'].created_at for oc in all_override_comments)
    newest_oc_created_at_text = newest_oc_created_at.strftime('%Y-%m-%d %H:%M UTC')
    oldest_oc_created_at_text = oldest_oc_created_at.strftime('%Y-%m-%d %H:%M UTC')
    report.write(textwrap.dedent(
    f"""

    ### At which rate are override commands issued? Who issues most of them?

    This section shows statistics derived from all override commands issued to
    date. The newest override command considered for this report was issued at
    {newest_oc_created_at_text}. The oldest override command was issued at
    {oldest_oc_created_at_text}.


    #### Override command rate over time

    In this plot all override commands are considered (no distinction is made
    based on the CI check name or JIRA ticket referred to in individual
    commands):
    """
    ))

    figure_file_abspath = plot_override_comment_rate_two_windows(all_override_comments)
    include_figure(
        report,
        figure_file_abspath,
        'Override comment rate plotted over time'
    )

    report.write(textwrap.dedent(
    """
    When you read the plot, ask yourself: do you see a trend? Are we getting
    better over time (does the overall override command rate decrease)?

    #### Override command rate over time resolved by most relevant JIRA tickets

    This plot shows the time evolution of the rate for override commands that
    refer to special JIRA tickets; those JIRA tickets that were referred to most
    often in all override commands (the instabilities that they represent have
    been dominant time sinks and pain points, and maybe still are):
    """
    ))
    counter = Counter([oc['ticket'] for oc in all_override_comments])
    top_ticketnames = [ticketname for ticketname, count in counter.most_common(10)]

    figure_file_abspath = plot_override_comment_rate_multiple_jira_tickets(
        all_override_comments, top_ticketnames)

    include_figure(
        report,
        figure_file_abspath,
        'Override comment rate plotted over time, resolved by individual JIRA tickets'
    )

    report.write(textwrap.dedent(
    """
    When you read the plot, ask yourself: did we address those instabilities
    that have hurt us most? Did we fix them properly, or are they coming back?

    #### People who issued most of the override commands

    Let us say THANKS to the people that help landing changes in DC/OS by
    constructing appropriate override commands:

    """
    ))

    topn = 15
    # report.write(f'\nTop {topn} override command issuer:\n\n')
    counter = Counter([oc['comment_obj'].user.login for oc in all_override_comments])
    tabletext = get_mdtable(
        ['GitHub login', 'Number of overrides'],
        [[item, count] for item, count in counter.most_common(topn)],
    )
    report.write(f'{tabletext}\n\n')

    # reportfragment = analyze_overrides(
    #     'Most frequent overrides (last 10 days)',
    #     10,
    #     all_override_comments,
    #     prs
    # )
    # report.write(reportfragment.getvalue())

    reportfragment = analyze_overrides(
        'Most frequent overrides (last 30 days)',
        30,
        all_override_comments,
        prs
    )
    report.write(reportfragment.getvalue())

    # reportfragment = analyze_overrides(
    #     'Most frequent overrides (all-time)',
    #     10**4,
    #     all_override_comments,
    #     prs
    # )
    # report.write(reportfragment.getvalue())


def analyze_overrides(heading, max_age_days, all_override_comments, prs):
    print(f'\n\n\n* Override comment analysis: {heading}')
    reportfragment = StringIO()
    reportfragment.write(f'\n\n### {heading}\n\n')
    reportfragment.write(f'Based on override commands issued in the last {max_age_days} days. ')
    analyze_overrides_last_n_days(all_override_comments, max_age_days, reportfragment)
    analyze_overrides_in_recent_prs(prs, max_age_days, reportfragment)

    # Find first occurrence of individual override JIRA tickets, and show the
    # ones that were used for the first time within the last N days (this
    # identifies new flakes).
    collector = defaultdict(list)
    for comment in all_override_comments:
        collector[comment['ticket']].append(comment['comment_obj'].created_at)

    reportfragment.write(textwrap.dedent(
    f"""

    JIRA tickets from override commands used for the first time within the last
    {max_age_days} days (these are new instabilities and they should probably be
    addressed!):\n\n

    """
    ))

    newtickets = []
    count = 0
    for ticket, created_dates in collector.items():
        earliest_date = min(created_dates)
        age = NOW - earliest_date
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            count += 1
            if count < 15:
                newtickets.append(ticket)
            else:
                # For long time frames (such as for all-time stats) this list
                # grows too big to be meaningful.
                newtickets.append('...')
                break

    # Add an initial space so that first JIRA ticket in enumeration will be
    # detected by the JIRA link filter.
    reportfragment.write(' ')
    reportfragment.write(', '.join(newtickets))
    reportfragment.write('\n\n')
    return reportfragment


def analyze_overrides_in_recent_prs(prs, max_age_days, reportfragment):
    """
    Find pull requests not older than `max_age_days` and extract all override
    commands issued in them. Perform a statistical analysis on this set of
    override commands.
    """
    print(f'** Histograms from override comments in PRs younger than {max_age_days} days')
    prs_to_analyze = []
    for pr in prs:
        # `pr.created_at` sadly is a native datetime object. It is known to
        # represent the time in UTC, however. `NOW` also is a datetime object
        # explicitly in UTC.
        age = NOW - pr.created_at
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            prs_to_analyze.append(pr)

    topn = 10
    print(f'   Top {topn} number of override commands issued per pull request:')
    counter = Counter([len(pr._override_comments) for pr in prs_to_analyze])
    tabletext = get_mdtable(
        ['Number of PRs', 'Number of overrides'],
        [[count, item] for item, count in counter.most_common(topn)]
    )
    print(tabletext)
    # Do not, for now, include this in the markdown report.


def analyze_overrides_last_n_days(override_comments, n, reportfragment):
    print(f'** Histograms from override comments younger than {n} days')
    max_age_days = n
    ocs_to_analyze = []
    for oc in override_comments:
        age = NOW - oc['comment_obj'].created_at
        if age.total_seconds() < 60 * 60 * 24 * max_age_days:
            ocs_to_analyze.append(oc)
    print(f'** Number of override comments: {len(ocs_to_analyze)}')
    reportfragment.write(f'Number of override commands issued: **{len(ocs_to_analyze)}**. ')
    oldest_created_at = min(c['comment_obj'].created_at for c in ocs_to_analyze)
    # `oldest_created_at` is a naive timezone object representing the time
    # of the comment creation in UTC. GitHub returns tz information, but PyGitHub
    # does not parse it properly. See
    # https://github.com/PyGithub/PyGithub/blob/365a0a24d3d2f06eeb4c93b4487fcfb88ae95dd0/github/GithubObject.py#L168
    # and https://github.com/PyGithub/PyGithub/issues/512 and
    # https://stackoverflow.com/a/30696682/145400.
    reportfragment.write(f'Oldest override command issued at {oldest_created_at} (UTC). ')
    build_histograms_from_ocs(ocs_to_analyze, reportfragment)


def build_histograms_from_ocs(override_comments, reportfragment):
    topn = 10
    print(f'   Top {topn} JIRA tickets used in override comments')
    reportfragment.write(
        f'\nTop {topn} JIRA tickets (do we work on the top ones? we should!):\n\n')

    counter = Counter([oc['ticket'] for oc in override_comments])
    tabletext = get_mdtable(
        ['JIRA ticket', 'Number of overrides'],
        [[item, count] for item, count in counter.most_common(topn)],
    )
    reportfragment.write(tabletext)

    print(f'   Top {topn} CI check names used in override comments')
    reportfragment.write(f'\nTop {topn} CI status check names:\n\n')

    counter = Counter([oc['checkname'] for oc in override_comments])
    tabletext = get_mdtable(
        ['Status check name', 'Number of overrides'],
        [[item, count] for item, count in counter.most_common(topn)],
    )
    reportfragment.write(tabletext)
    reportfragment.write('\n\n')


def get_mdtable(header_list, value_matrix):
    """
    Generate table text in Markdown.
    """
    assert value_matrix

    tw = pytablewriter.MarkdownTableWriter()
    tw.stream = StringIO()
    tw.header_list = header_list
    tw.value_matrix = value_matrix
    # Potentially use
    # writer.align_list = [Align.LEFT, Align.RIGHT, ...]
    # see https://github.com/thombashi/pytablewriter/issues/2
    tw.margin = 1
    tw.write_table()
    # print(textwrap.indent(tw.stream.getvalue(), '    '))
    return tw.stream.getvalue()


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
    # comments emitted straight in the main commentary thre analyze_pr_comments(pad.
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
            log.info(
                'Invalid override command: slash in ticket name '
                '(wrong order of args?): %s',
                ticket
            )
            return None

        if not re.match('^[A-Z_]+-[0-9]+$', ticket):
            log.info('Invalid override command: ticket does not match regex: %s', ticket)
            return None

        checkname = match.group('checkname').strip()
        if len(checkname.split()) > 1:
            log.info('Invalid override command: whitespace in checkname: `%s`', checkname)
            return None

        if linecount > 1:
            ...
            # log.info('Mergebot override in multi-line comment:\n%r', text)

        # Create the override comment data structure used elsewhere in the
        # program.
        override_comment = {
            'prnumber': pr.number,
            'checkname': checkname,
            'ticket': ticket,
            'comment_obj': comment
        }

        return override_comment
    return None


def calc_override_comment_rate(
        override_comments, window_width_days_1=3, window_width_days_2=14):

    # Rolling window of N days width. The column does not matter, only evaluate
    # number of events (rows) in the rolling window, and count them, then
    # normalize.

    # pygithub returns naive datetime objects, but we know that the
    # corresponding timezone is UTC. The `columncount` column's name will only
    # be really important / meaningful after resampling below with the
    # subsequent `count()` aggregation.
    df_raw = pd.DataFrame(
        {'commentcount': [1 for c in override_comments]},
        index=[
            pd.Timestamp(c['comment_obj'].created_at, tz='UTC') for
                c in override_comments
        ]
    )

    # Sort by time (comment creation time).
    df_raw.sort_index(inplace=True)

    # Example head, tail
    #                            commentcount
    # 2017-11-23 01:07:39+00:00             1
    # 2017-11-23 01:10:34+00:00             1
    # 2017-11-23 01:57:27+00:00             1
    # 2017-11-23 02:11:21+00:00             1
    # 2017-11-27 17:07:51+00:00             1
    # [...]
    # 2018-11-23 09:22:01+00:00             1
    # 2018-11-23 15:08:51+00:00             1
    # 2018-11-23 15:26:34+00:00             1
    # 2018-11-23 16:41:16+00:00             1
    # 2018-11-23 16:41:25+00:00             1

    # Resample so that there is one data point per hour (this downsamples
    # sometimes when multiple commands were issued within the same hour, and
    # upsamples during other times, such as night time and holiday time). Count
    # number of values within each bucket (the number of override commands per
    # hour). For those hours where there is no data point at all `count()`
    # results in a value of 0. Add an offset of 0.5 hours so that the data
    # points are actually e.g. `2017-12-17 12:30:00` instead of `2017-12-17
    # 12:00:00`. Otherwise there would be a systematic time shift in the data
    # after resampling.
    df_resampled = df_raw.resample('1H').count()

    # This is how the example from above looks now.
    #                            commentcount
    # 2017-11-23 01:00:00+00:00             3
    # 2017-11-23 02:00:00+00:00             1
    # 2017-11-23 03:00:00+00:00             0
    # 2017-11-23 04:00:00+00:00             0
    # 2017-11-23 05:00:00+00:00             0
    # [...]
    # 2018-11-23 12:00:00+00:00             0
    # 2018-11-23 13:00:00+00:00             0
    # 2018-11-23 14:00:00+00:00             0
    # 2018-11-23 15:00:00+00:00             2
    # 2018-11-23 16:00:00+00:00             2

    # Make it so that the time series contains data until today, well, now,
    # showing zeros as a value. For getting there first add a data point with a
    # timestamp from today, with the value 0. Then perform another resample
    # operation with the same interval as before, and fill NaN values with
    # zeros.
    df_resampled.loc[pd.Timestamp.utcnow()] = [0]

    # Example with added timestamp, tail:
    #                                   commentcount
    # ...
    # 2018-11-23 13:00:00+00:00                    0
    # 2018-11-23 14:00:00+00:00                    0
    # 2018-11-23 15:00:00+00:00                    2
    # 2018-11-23 16:00:00+00:00                    2
    # 2018-11-24 11:53:31.995233+00:00             0

    df_resampled = df_resampled.resample('1H').asfreq().fillna(0)

    # This is how the example looks like now.
    #                            commentcount
    # 2017-11-23 01:00:00+00:00           3.0
    # 2017-11-23 02:00:00+00:00           1.0
    # 2017-11-23 03:00:00+00:00           0.0
    # 2017-11-23 04:00:00+00:00           0.0
    # 2017-11-23 05:00:00+00:00           0.0
    # [...]
    # 2018-11-24 07:00:00+00:00           0.0
    # 2018-11-24 08:00:00+00:00           0.0
    # 2018-11-24 09:00:00+00:00           0.0
    # 2018-11-24 10:00:00+00:00           0.0
    # 2018-11-24 11:00:00+00:00           0.0

    # Note(JP): if a resolution of 1 hour is ever a performance problem
    # then decrease resolution to one day:
    # df_resampled = df_raw.resample('1D', loffset='12H').count()

    # Apply rolling time window analysis, and `sum()` the values within a window
    # because as of here the value per data points represents the number of
    # override commands issued per day.
    rollingwindow_1 = df_resampled['commentcount'].rolling(
        window='%sD' % window_width_days_1,
        min_periods=0
    )
    commentrate_1 = rollingwindow_1.sum() / float(window_width_days_1)

    # In the resulting Series object, the request rate value is assigned to the
    # right window boundary index value (i.e. to the newest timestamp in the
    # window). For presentation it is more convenient to have it assigned
    # (approximately) to the temporal center of the time window. That makes
    # sense for intuitive data interpretation of a single rolling window time
    # series, but is essential for meaningful presentation of multiple rolling
    # window series in the same plot (when their window width varies). Invoking
    # `rolling(..., center=True)` however yields `NotImplementedError: center is
    # not implemented for datetimelike and offset based windows`. As a
    # workaround, shift the data by half the window size to 'the left': shift
    # the timestamp index by a constant / offset.
    # Add an epsilon (one second) for working around a plotting bug
    # https://github.com/pandas-dev/pandas/issues/22586
    offset_seconds = - int(window_width_days_1 * 24 * 60 * 60 / 2.0) + 1
    commentrate_1 = commentrate_1.shift(offset_seconds, freq='s')

    # In the resulting time series, all leftmost values up to the rolling window
    # width are dominated by the effect that the rolling window (incoming from
    # the left) does not yet completely overlap with the data. That is, here the
    # rolling window result is (linearly increasing) systematically to small.
    # Because by now the time series has one sample per day, the number of
    # leftmost samples with a bad result corresponds to the window width in
    # days. Return just the slice `[window_width_days:]`. The same consideration
    # does not hold true for the right end of the data: the window is rolled not
    # further than the right end of the data.
    commentrate_1 = commentrate_1[window_width_days_1:]

    # Same thing for a more wide rolling window.
    rollingwindow_2 = df_resampled['commentcount'].rolling(
        window='%sD' % window_width_days_2,
        min_periods=0
    )
    commentrate_2 = rollingwindow_2.sum() / float(window_width_days_2)

    # Add an epsilon (one second) for working around a plotting bug
    # https://github.com/pandas-dev/pandas/issues/22586
    offset_seconds = - int(window_width_days_2 * 24 * 60 * 60 / 2.0) + 1
    commentrate_2 = commentrate_2.shift(offset_seconds, freq='s')
    commentrate_2 = commentrate_2[window_width_days_2:]

    return commentrate_1, window_width_days_1, commentrate_2, window_width_days_2, df_raw, df_resampled


def plot_override_comment_rate_two_windows(override_comments):

    plt.figure()

    commentrate_1, window_width_days_1, commentrate_2, window_width_days_2, _, _ = \
        calc_override_comment_rate(override_comments)

    ax = commentrate_1.plot(
        linestyle='solid',
        color='gray',
    )

    commentrate_2.plot(
        linestyle='solid',
        marker='None',
        color='black',
        ax=ax
    )

    # The legend story is shitty with pandas intertwined w/ mpl.
    # http://stackoverflow.com/a/30666612/145400
    ax.legend([
        f'rolling window average ({window_width_days_1} days)',
        f'rolling window average ({window_width_days_2} days)'
        ],
        numpoints=4
    )

    plt.xlabel('Time')
    plt.ylabel('Override command rate [1/day]')
    # set_title('Override command rate (from both DC/OS repositories)')
    # set_subtitle('Arithmetic mean over rolling time window')
    # plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.tight_layout()
    return savefig('Override command rate')


def plot_override_comment_rate_multiple_jira_tickets(
        all_override_comments, ticketnames):

    # Create a new figure and an Axes object. Plot all line plots into the same
    # Axes object.
    fig = plt.figure()
    ax = fig.gca()

    for ticketname in ticketnames:
        ocs = [oc for oc in all_override_comments if oc['ticket'] == ticketname]
        commentrate_1, window_width_days_1, commentrate_2, window_width_days_2, _, _ = \
            calc_override_comment_rate(ocs)

        commentrate_2.plot(
            linestyle='solid',
            marker='None',
            markersize=4,
            ax=ax,
            )

    ax.legend(
        [f'{ticketname}' for ticketname in ticketnames],
        numpoints=4,
        loc='upper left'
    )

    plt.xlabel('Time')
    plt.ylabel('Override command rate [1/day]')
    # set_title('Override command rate for most frequently tagged JIRA tickets')
    # set_subtitle('Arithmetic mean over rolling time window')
    # plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.tight_layout()
    return savefig('Override command rate for top JIRA tickets')


def savefig(title):
    """
    Save figure file to `OUTDIR`.

    Expected to return just the base name (not the complete path).
    """
    # Lowercase, replace special chars with whitespace, join on whitespace.
    cleantitle = '-'.join(re.sub('[^a-z0-9]+', ' ', title.lower()).split())

    fname = TODAY + '_' + cleantitle

    fpath_figure = os.path.join(OUTDIR, fname + '.png')
    log.info('Writing PNG figure to %s', fpath_figure)
    plt.savefig(fpath_figure, dpi=150)
    return os.path.basename(fpath_figure)


class LabelType(Enum):
    SHIP_IT = 1
    READY_FOR_REVIEW = 2


class PRLabel:
    """
    Represent a pull request label. It has a type and a creation time. Use only
    the label type when comparing with other PRLabel instances. This simplifies
    analyses such as building a label transition histogram and finding a certain
    label type in a list of labels.
    """
    __slots__ = 'created_at', 'type'

    def __init__(self, created_at):
        self.created_at = created_at

    def __hash__(self):
        return hash(self.type)

    def __eq__(self, other):
        return self.type is other.type

    def __repr__(self):
        """
        Return the label type w/o the 'LabelType.' prefix.
        """
        return str(self.type)[10:]


class ShipIt(PRLabel):
    type = LabelType.SHIP_IT


class ReadyForReview(PRLabel):
    type = LabelType.READY_FOR_REVIEW


def pr_analyze_label_transitions(pr):
    """
    Analyze evolution of labels set for this pull request.

    Extract the following metrics:

        - time from opening PR to last ship it label
        - time from opening PR to last ready for review label
        - time from last ship it label to merge
        - time from last ready for review label to merge
        - time from last ready for review label to last ship it label

    Store these metrics on the `pr` object in a special attribute, a dictionary.

    Return tuple of PR labels, in order, until merge. Examples:

       (READY_FOR_REVIEW, SHIP_IT)
    or (READY_FOR_REVIEW,)
    or ()
    or (SHIP_IT,)
    or (READY_FOR_REVIEW, READY_FOR_REVIEW)
    or (READY_FOR_REVIEW, SHIP_IT, READY_FOR_REVIEW, SHIP_IT)

    This only considers SHIP IT and READY FOR REVIEW labels (effectively
    ignored all other labels).
    """

    if not hasattr(pr, '_events'):
        log.warning('PR object does not have _events property: %r', pr)
        return ()

    labels_in_order_until_merge = []

    for event in pr._events:
        if event.event == 'labeled':
            # The deserialization into `event.label.name` seems to be buggy
            # with pygithub 1.43, failing for a small number of events with
            # an AttributeError, despite raw data being there. I filed
            # https://github.com/PyGithub/PyGithub/issues/991
            #
            # Only account for label changes until merge.
            if event.created_at < pr.merged_at:

                if 'ship' in event._rawData['label']['name'].lower():
                    labels_in_order_until_merge.append(
                        ShipIt(event.created_at))

                elif 'review' in event._rawData['label']['name'].lower():
                    labels_in_order_until_merge.append(
                        ReadyForReview(event.created_at))

    # Do not rely on labels to already be in chronological order. Default is to
    # sort in ascending order, with events that happened later in time appearing
    # later in the list.
    labels_in_order_until_merge.sort(key=lambda x: x.created_at)

    # Prepare a dictionary for collecting various time metrics of this pull
    # request. Prepopulate the values with `None` because not all metrics can
    # be found for all pull requests.
    ts = {
        'time_pr_open_to_last_shipit': None,
        'time_pr_open_to_last_rfr': None,
        'time_last_shipit_to_pr_merge': None,
        'time_last_rfr_to_pr_merge': None,
        'time_last_rfr_to_last_shipit': None
    }

    last_shipit_time = None

    for label in reversed(labels_in_order_until_merge):

        if label.type is LabelType.SHIP_IT:
            ts['time_pr_open_to_last_shipit'] = label.created_at - pr.created_at
            ts['time_last_shipit_to_pr_merge'] = pr.merged_at - label.created_at
            last_shipit_time = label.created_at

        if label.type is LabelType.READY_FOR_REVIEW:
            ts['time_pr_open_to_last_rfr'] = label.created_at - pr.created_at
            ts['time_last_rfr_to_pr_merge'] = pr.merged_at - label.created_at
            if last_shipit_time:
                ts['time_last_rfr_to_last_shipit'] = last_shipit_time - label.created_at

    # Modify PR in place.
    pr._label_transition_timings = ts

    return tuple(labels_in_order_until_merge)


def analyze_merged_prs(prs, report):

    log.info('Filter merged pull requests.')
    filtered_prs = [pr for pr in prs if pr.merged_at is not None]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    # Hopefully helpful note: `pr.head.label` is e.g.
    # `mesosphere:mergebot/dcos/1.11/3292` for a pull request created in
    # mesosphere/dcos-enterprise created by Mergebot as of a bump-ee command.

    # Proceed with analyzing only those pull requests that were not created by
    # mergebot. This ignores an important class of pull request: all downstream
    # PRs created via the bump-ee command. This is intentional, following the
    # idea that a pull request pair comprised of an upstream PR with the
    # corresponding downstream PR is tracking one unit of change to DC/OS.
    log.info('Filter pull requests not created by mergebot.')
    filtered_prs = [pr for pr in filtered_prs if 'mergebot' not in pr.user.login]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    log.info('Filter pull requests not having `Train` or `train` in title.')
    filtered_prs = [pr for pr in filtered_prs if 'train' not in pr.title.lower()]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    # Future goal is to distinguish PR types, to look at train PRs separately,
    # and to filter PRs by certain criteria in general, such as
    # - how many lines do they change
    # - are these just simple package bumps?
    # - ...

    log.info('Analyze label transitions in filtered PRs')
    label_transitions = [pr_analyze_label_transitions(pr) for pr in filtered_prs]

    log.info('Pull request label transition histogram, top 50')
    counter = Counter(label_transitions)
    for transition, count in counter.most_common(50):
        print('{:>8}: {}'.format(count, transition))

    log.info('Build main Dataframe')
    df_dict = {
        'created_at': [pr.created_at for pr in filtered_prs],
        'openseconds': [
            (pr.merged_at - pr.created_at).total_seconds() for
            pr in filtered_prs],
        }

    # All these metrics are time differences measured in seconds.
    time_diff_metrics = (
        'time_pr_open_to_last_shipit',
        'time_pr_open_to_last_rfr',
        'time_last_shipit_to_pr_merge',
        'time_last_rfr_to_pr_merge',
        'time_last_rfr_to_last_shipit'
    )
    for metric in time_diff_metrics:
        df_dict[metric + '_seconds'] = [
            pr._label_transition_timings[metric].total_seconds()
            if pr._label_transition_timings[metric] is not None else np.NaN for
            pr in filtered_prs]

    # I think the point in time when a pull request has been merged is the
    # better reference for determining metrics like throughput and latency than
    # the point in time when a pull request has been created.
    df = pd.DataFrame(
        df_dict,
        index=[pd.Timestamp(pr.merged_at) for pr in filtered_prs]
        )
    # Sort by time.
    df.sort_index(inplace=True)

    # Convert a number of time differences measured in seconds to days, for
    # easier human consumption in plots.
    df['time_pr_open_to_merge_days'] = df['openseconds'] / 86400


    for metric in time_diff_metrics:
        df[metric + '_days'] = df[metric + '_seconds'] / 86400

    log.info('Main Dataframe:')
    print(df)

    # This line assumes that somewhere in the code path a figure has been
    # created before, now create a fresh one.
    plt.figure()

    latency_median, \
    figure_filepath_latency_raw_linscale, \
    figure_filepath_latency_raw_logscale = plot_latency(
        df, 'time_pr_open_to_merge_days')

    plt.figure()
    throughput_mean, figure_throughput_filepath = plot_throughput(filtered_prs)

    plt.figure()
    quality = throughput_mean / latency_median
    df['quality'] = quality
    figure_quality_filepath = plot_quality(df)

    plt.figure()
    figure_latency_focus_on_mean = plot_latency_focus_on_mean(
        df, 'time_pr_open_to_merge_days')

    # Create plots with a different TTM metric, the time difference
    # between the last ship it label and the PR merge. This applies to
    # significantly less pull requests, especially in the more distant past.
    plt.figure()
    figure_filepath_ttm_shipit_to_merge_focus_on_mean = \
        plot_latency_focus_on_mean(
            df['2017-03-01':],
            'time_last_shipit_to_pr_merge_days'
        )

    plt.figure()
    _, \
    figure_filepath_ttm_shipit_to_merge_raw_linscale, \
    figure_filepath_ttm_shipit_to_merge_raw_logscale = plot_latency(
        df['2017-03-01':], 'time_last_shipit_to_pr_merge_days', show_mean=False)

    plt.figure()
    figure_filepath_various_latencies = plot_pr_lifecycle_latency_metrics(
        df['2017-07-01':])

    report.write(textwrap.dedent(
    """

    ## Pull request (PR) integration velocity: time-to-merge (TTM)

    This analysis considers merged DC/OS pull requests across the two DC/OS
    repositories ("[upstream](https://github.com/dcos/dcos)" and
    "[downstream](https://github.com/mesosphere/dcos-enterprise)"). The goal is
    to make the analysis represent how individual developers perceive the
    process which is why Mergebot-created pull requests and manually created
    merge train pull requests are _not_ considered. A pull request pair
    (comprised of an upstream PR plus its corresponding Mergebot-managed
    downstream PR) is counted as a single pull request.


    ### Time from opening the PR to merge

    The following plot shows the number of days it took for individual PRs to
    get merged. Each dot represents a single merged PR (or PR pair). The black
    and orange lines show the median and arithmetic mean, correspondingly,
    averaged over a rolling time window of 14 days width.
    """
    ))

    include_figure(
        report,
        figure_filepath_latency_raw_linscale,
        'Pull request integration latency'
    )

    report.write(textwrap.dedent(
    """

    As of outliers this plot is hard to resolve in the details. Let's look at
    the same graph with a logarithmic scale on the latency axis:
    """
    ))

    include_figure(
        report,
        figure_filepath_latency_raw_logscale,
        'Pull request integration latency (logarithmic scale)'
    )

    report.write(textwrap.dedent(
    """
    The latency values are usually spread across about four orders of magnitude
    at any given time, with no uniform density distribution. There is tendency
    for cluster formation. Neither the mean nor the median represent the raw
    data well. Much can be understood by looking at the distribution of the raw
    data points, ignoring mean and median.

    When you read the above plot ask yourself: does the latency appear to be in
    a tolerable regime? Do you see a trend? Does the raw data appear to be
    clustered? How do the clusters evolve?

    The following plot, instead of showing the raw data, focuses on showing the
    mean and median and -- to quantify the overall degree of scattering --
    additionally visualizes the standard deviation of the data.

    """
    ))

    include_figure(
        report,
        figure_latency_focus_on_mean,
        'Pull request integration latency (focus on mean)'
    )

    report.write(textwrap.dedent(
    """

    ### PR life cycle resolved in detail (ship-it to merge, etc)

    A subset of the merged pull requests went through a proper life cycle which
    requires a "ship it" label to be set on the pull request before merging. For
    PRs which fulfill this criterion the following plot shows the time
    difference between the last applied ship it label and the merge time (note
    that the time window shown in the plots below starts around March 2017, as
    opposed to May 2016 above -- the ship-it label concept was introduced only
    in 2017).
    """
    ))

    # include_figure(
    #     report,
    #     figure_filepath_ttm_shipit_to_merge_raw_linscale,
    #     'Pull request TTM ship-it-to-merge (linear scale)'
    # )

    # include_figure(
    #     report,
    #     figure_filepath_ttm_shipit_to_merge_focus_on_mean,
    #     'Pull request TTM ship-it-to-merge (focus on mean)'
    # )

    include_figure(
        report,
        figure_filepath_ttm_shipit_to_merge_raw_logscale,
        'Pull request TTM ship-it-to-merge (logarithmic scale)'
    )

    report.write(textwrap.dedent(
    """
    The time a pull request spent between ship-it and merge should be
    insignificant relative to time spent in previous stages of its life cycle.
    After all, a ship-it label indicates that fellow developers found that the
    change is good to go.

    In other words, the total time of a PR spent between opening and merge
    should be dominated by the previous two stages. Whether or not this is the
    case is shown by the following graph (linear scale without the raw data,
    showing only the rolling window median):
    """
    ))

    include_figure(
        report,
        figure_filepath_various_latencies,
        'Pull request latencies, various metrics'
    )


    report.write(textwrap.dedent(
    """

    ## Pull request integration velocity: Throughput

    The following plot shows the number of PRs merged per day, averaged over a
    rolling time window of two weeks width.
    """
    ))

    include_figure(
        report,
        figure_throughput_filepath,
        'Pull request integration throughput'
    )

    # report.write(textwrap.dedent(
    # """

    # ### Velocity

    # The greater the throughput the better, the smaller the latency the better.
    # Following this rationale the following plot shows the "velocity" defined
    # as throughput divided by latency.
    # """
    # ))

    # include_figure(
    #     report,
    #     figure_quality_filepath,
    #     'Pull request integration velocity'
    # )


def include_figure(report, filepath, heading):
    report.write(f'\n\n[![{heading}]({filepath} "{heading}")]({filepath})\n\n')


# What is good is low time to merge, and many pull requests merged per time.
# So, divide throughput by latency.


def plot_quality(df):
    df['quality'].plot(color='red')
    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day] / TTM [day]')
    # set_title('PR integration velocity for PRs in both DC/OS repos')
    # subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #     matcher.subtitle
    # set_subtitle('Raw data')
    #plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.tight_layout()
    return savefig('Pull request integration velocity')


def plot_throughput(filtered_prs):

    # Rolling window of N days width. The column does not matter, only evaluate
    # number of events (rows) in the rolling window, and count them.

    df = pd.DataFrame(
        {
            'foo': [1 for pr in filtered_prs]
        },
        index=[pd.Timestamp(pr.merged_at) for pr in filtered_prs]
        )

    # Sort by time (when the PRs have been merged).
    df.sort_index(inplace=True)

    rollingwindow = df['foo'].rolling('14d')
    throughput = rollingwindow.count()/14.0
    # stddev = rollingwindow.std()

    ax = throughput.plot(
        linestyle='solid',
        color='black',
    )

    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day]')

    ax.legend([
        f'rolling window mean (14 days)',
        ],
        numpoints=4
    )

    # plt.tight_layout(rect=(0, 0, 1, 95))
    plt.tight_layout()

    return throughput, savefig('Pull request integration throughput')


def _plot_latency_core(df, metricname, show_mean=True):
    ax = df[metricname].plot(
        # linestyle='dashdot',
        linestyle='None',
        color='gray',
        marker='.',
        markersize=4,
        markeredgecolor='gray'
    )
    plt.xlabel('Pull request merge time')
    plt.ylabel('Latency [days]')
    #set_title('Time-to-merge for PRs in both DC/OS repositories')
    # subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #    matcher.subtitle
    #set_subtitle('Raw data')
    #plt.tight_layout(rect=(0, 0, 1, 0.95))

    rollingwindow = df[metricname].rolling('14d')
    mean = rollingwindow.mean()
    median = rollingwindow.median()

    median.plot(
        linestyle='solid',
        dash_capstyle='round',
        color='black',
        linewidth=1.3,
        ax=ax
    )

    if show_mean:
        mean.plot(
            linestyle='solid',
            color='#e05f4e',
            linewidth=1.3,
            ax=ax
        )

    legendlist = [
        f'individual PRs',
        f'rolling window median (14 days)',
        ]

    if show_mean:
        legendlist.append(f'rolling window mean (14 days)')

    ax.legend(legendlist,numpoints=4)
    return median, ax


def plot_latency(df, metricname, show_mean=True):

    median, ax = _plot_latency_core(df, metricname, show_mean)
    plt.tight_layout()
    figure_filepath_latency_raw_linscale = savefig(
        f'PR integration latency (linear scale), metric: {metricname}')

    median, ax = _plot_latency_core(df,  metricname, show_mean)
    ax.set_yscale('log')
    plt.tight_layout()
    figure_filepath_latency_raw_logscale = savefig(
        f'PR integration latency (logarithmic scale), metric:  {metricname}')

    return (
        median,
        figure_filepath_latency_raw_linscale,
        figure_filepath_latency_raw_logscale
    )


def plot_latency_focus_on_mean(df, metricname):

    rollingwindow = df[metricname].rolling('14d')
    mean = rollingwindow.mean()
    ax = mean.plot(
        linestyle='solid',
        color='#e05f4e',
        linewidth=1.3,
    )

    median = rollingwindow.median()

    ax = median.plot(
        linestyle='solid',
        color='black',
        linewidth=1.5,
    )

    stddev = rollingwindow.std()
    upperbond = mean + stddev
    lowerbond = mean - stddev

    ax.fill_between(
        mean.index,
        lowerbond,
        upperbond,
        facecolor='gray',
        alpha=0.3
    )

    # With the type of data at hand here the global maximum of the median is
    # expected to be lower than the global maximum of the mean; and we are not
    # that interested in seeing the global max of the mean in the plot as it is
    # quite sensitive to outliers.
    plt.ylim((-0.5, median.max() + 0.1 * median.max()))

    plt.xlabel('Pull request merge time')
    plt.ylabel('Latency [days]')
    # plt.tight_layout(rect=(0, 0, 1, 0.95))

    ax.legend([
        f'rolling window mean (14 days)',
        f'rolling window median (14 days)',
        f'rolling window std dev (14 says)',
        ],
        numpoints=4,
        loc='upper left'
    )

    plt.tight_layout()
    return savefig(f'PR integration latency focus on mean, metric: {metricname}')


def plot_pr_lifecycle_latency_metrics(df):

    # Plot `time_pr_open_to_merge` and then those three metrics that add up to
    # this (in case of the ideal intended PR life cycle), to see the individual
    # contributions.
    metricnames = (
        #'time_pr_open_to_merge',
        #'time_pr_open_to_last_shipit',
        'time_pr_open_to_last_rfr',
        'time_last_rfr_to_last_shipit',
        'time_last_shipit_to_pr_merge',
        #'time_last_rfr_to_pr_merge',
    )

    for metricname in metricnames:
        rollingwindow = df[metricname + '_days'].rolling('14d')
        median = rollingwindow.median()
        ax = median.plot(
            linestyle='solid',
            linewidth=1.5,
        )

    # For the time range 2017-05 to 2018-12 it makes sense to cut the graph
    # at 12 so that the details are easier to resolve.
    #plt.ylim((-0.5, 12))
    plt.xlabel('Pull request merge time')
    plt.ylabel('Latency [days], 14 day rolling window median')
    # plt.tight_layout(rect=(0, 0, 1, 0.95))

    # legend_entries = [f'{mn} rolling window median (14 days)' for mn in metricnames]

    ax.legend([
        'Stage 1: PR opened -> Ready for Review',
        'Stage 2: Ready for Review -> Ship It',
        'Stage 3: Ship It -> PR merged'
        ],
        numpoints=4,
        loc='upper left'
    )

    plt.tight_layout()
    return savefig(f'PR integration latency, various metrics')


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
    matplotlib.rcParams['figure.figsize'] = [10.0, 6.5]
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 150
    # mpl.rcParams['font.size'] = 12

    original_color_cycle = matplotlib.rcParams['axes.prop_cycle']

    plt.style.use('ggplot')

    # ggplot's color cylcle seems to be too short for having 8 line plots on the
    # same Axes.
    matplotlib.rcParams['axes.prop_cycle'] = original_color_cycle


def load_prs_from_file(filepath):
    log.info('Unpickle from file: %s', filepath)
    with open(filepath, 'rb') as f:
        return pickle.loads(f.read())


if __name__ == "__main__":
    main()
