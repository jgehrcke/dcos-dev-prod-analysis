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
import textwrap

from io import StringIO
from collections import Counter, defaultdict
from datetime import datetime

import pytablewriter
import pandas as pd
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
        'dcos-enterprise_pull-requests-with-comments.pickle')

    prs_upstream = load_prs_from_file(
        'dcos_pull-requests-with-comments.pickle')

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
    prs_for_comment_analysis = [
        pr for pr in
        itertools.chain(prs_downstream.values(), prs_upstream.values())
    ]

    newest_pr_created_at = max(pr.created_at for pr in prs_for_comment_analysis)
    newest_pr_created_at_text = newest_pr_created_at.strftime('%Y-%m-%d %H:%M UTC')

    matplotlib_config()

    now_text = NOW.strftime('%Y-%m-%d %H:%M UTC')
    markdownreport = StringIO()
    markdownreport.write(textwrap.dedent(
    f"""
    % DC/OS developer productivity report
    %
    % Generated on {now_text}

    The report is generated based on GitHub pull request data from both DC/OS
    repositories. The code for generating this report lives in
    [`dcos-dev-prod-analysis`](https://github.com/jgehrcke/dcos-dev-prod-analysis).
    The newest pull request considered for this report was created at
    {newest_pr_created_at_text}.


    """
    ).strip())

    analyze_pr_comments(prs_for_comment_analysis, markdownreport)

    prs_for_throughput_analysis = prs_for_comment_analysis

    analyze_merged_prs(prs_for_throughput_analysis, markdownreport)

    log.info('Rewrite JIRA ticket IDs in the Markdown report')
    report_md_text = markdownreport.getvalue()
    # Match JIRA ticket string only when there is a leading space
    report_md_text = re.sub(
        " (?P<ticket>[A-Z_]+-[0-9]+)", " [\g<ticket>](https://jira.mesosphere.com/browse/\g<ticket>)",
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

    ## Status check override report (CI instability)

    An override command associates the name of a failed CI check with a JIRA
    ticket tracking the specific cause or symptom of the problem.

    Example: when a developer issues the override command
    ```
    override-status teamcity/dcos/test/dcos-e2e/docker/static/strict https://jira.mesosphere.com/browse/DCOS_OSS-2115

    ```
    they intend to express that the CI check titled
    `teamcity/dcos/test/dcos-e2e/docker/static/strict` (which itself is
    comprised of hundreds of individual tests) failed as of a known instability
    in a specific test called `test_vip` (the details are to be inferred from
    the corresponding JIRA ticket).

    Every single override command

    - was issued by a human after they have carefully analyzed the specific
      cause or symptom of the problem.
    - means that the corresponding developer spent significant time (minutes
      to hours) figuring out the correct override command.

    That is, override command data is directly representing the pain individual
    developers experience when they attempt to land patches in the two DC/OS
    repositories.
    """
    ))

    # Identify and leave note about newest override command.
    newest_oc_created_at = max(oc['comment_obj'].created_at for oc in all_override_comments)
    oldest_oc_created_at = min(oc['comment_obj'].created_at for oc in all_override_comments)
    newest_oc_created_at_text = newest_oc_created_at.strftime('%Y-%m-%d %H:%M UTC')
    oldest_oc_created_at_text = oldest_oc_created_at.strftime('%Y-%m-%d %H:%M UTC')
    report.write(f'The newest override command considered for this report was issued at {newest_oc_created_at_text}.')
    report.write(f' The oldest override command was issued at {oldest_oc_created_at_text}.')

    report.write(textwrap.dedent(
   f"""

    ### All-time override commands

    This section shows statistics derived from all override commands issued to
    date.

    """
    ))

    topn = 10
    report.write(f'\nTop {topn} override command issuer:\n\n')
    counter = Counter([oc['comment_obj'].user.login for oc in all_override_comments])
    tabletext = get_mdtable(
        ['GitHub login', 'Number of overrides'],
        [[item, count] for item, count in counter.most_common(topn)],
    )
    report.write(f'{tabletext}\n\n')

    figure_file_abspath = plot_override_comment_rate_two_windows(all_override_comments)
    include_figure(
        report,
        figure_file_abspath,
        'Override comment rate plotted over time'
    )

    counter = Counter([oc['ticket'] for oc in all_override_comments])
    top_ticketnames = [ticketname for ticketname, count in counter.most_common(10)]
    figure_file_abspath = plot_override_comment_rate_multiple_jira_tickets(
        all_override_comments, top_ticketnames)

    include_figure(
        report,
        figure_file_abspath,
        'Override comment rate plotted over time, resolved by individual JIRA tickets'
    )

    reportfragment = analyze_overrides(
        'Most frequent overrides (last 10 days)',
        10,
        all_override_comments,
        prs
    )
    report.write(reportfragment.getvalue())

    reportfragment = analyze_overrides(
        'Most frequent overrides (last 30 days)',
        30,
        all_override_comments,
        prs
    )
    report.write(reportfragment.getvalue())

    reportfragment = analyze_overrides(
        'Most frequent overrides (all-time)',
        10**4,
        all_override_comments,
        prs
    )
    report.write(reportfragment.getvalue())


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
    reportfragment.write(f'JIRA tickets from override commands used for the first time within the last {max_age_days} days:\n\n')

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
    reportfragment.write(f'\nTop {topn} JIRA tickets:\n\n')

    counter = Counter([oc['ticket'] for oc in override_comments])
    tabletext = get_mdtable(
        ['JIRA ticket', 'Number of overrides'],
        [[item, count] for item, count in counter.most_common(topn)],
    )
    reportfragment.write(tabletext)

    print(f'   Top {topn} CI check names used in override comments')
    reportfragment.write(f'\nTop {topn} status keys (check names):\n\n')

    counter = Counter([oc['checkname'] for oc in override_comments])
    tabletext = get_mdtable(
        ['Status check', 'Number of overrides'],
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
            log.info('Invalid override command: slash in ticket name (wrong order of args?): %s', ticket)
            return None

        if not re.search('[A-Z_]+-[0-9]+', ticket):
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


def calc_override_comment_rate(override_comments):

    # Rolling window of one week width. The column does not matter, only
    # evaluate number of events (rows) in the rolling window, and count them,
    # then normalize.

    df_raw = pd.DataFrame(
        {'foo': [1 for c in override_comments]},
        index=[pd.Timestamp(c['comment_obj'].created_at) for c in override_comments]
    )

    # Sort by time (comment creation time).
    df_raw.sort_index(inplace=True)

    # Resample so that there is one data point per hour (this downsamples
    # sometime when multiple commands were issued within the same hour, and
    # upsamples during other times, such as night time and holiday time). Count
    # number of values within each bucket (the number of override commands per
    # hour). For those hours where there is no data point at all `count()`
    # results in a value of 0. Add an offset of 0.5 hours so that the data
    # points are actually e.g. `2017-12-17 12:30:00` instead of `2017-12-17
    # 12:00:00`. Otherwise there would be a systematic time shift in the data
    # after resampling.
    df_resampled = df_raw.resample('1H').count()
    # Note(JP): if this a resolution of 1 hour is ever a performance problem
    # then decrease resolution to one day:
    # df_resampled = df_raw.resample('1D', loffset='12H').count()

    # Example for how the dataframe can look like here:
    #
    # After resample().count() on df:
    #             foo
    # 2017-12-17    3
    # 2017-12-18    6
    # 2017-12-19   26
    # 2017-12-20   15
    # 2017-12-21   10
    # 2017-12-22    4
    # 2017-12-23    0
    # 2017-12-24    0
    # 2017-12-25    0
    # 2017-12-26    2
    # 2017-12-27    0
    # 2017-12-28    4
    # 2017-12-29    0
    # 2017-12-30    0

    # Apply rolling time window analysis, and `sum()` the values within a window
    # because as of here the value per data points represents the number of
    # override commands issued per day.
    window_width_days_1 = 3
    rollingwindow_1 = df_resampled['foo'].rolling(
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
    window_width_days_2 = 14
    rollingwindow_2 = df_resampled['foo'].rolling(
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
        f'{window_width_days_1} d window',
        f'{window_width_days_2} d window'
        ],
        numpoints=4
    )

    plt.xlabel('Time')
    plt.ylabel('Override command rate [1/day]')
    set_title('Override command rate (from both DC/OS repositories)')
    set_subtitle('Arithmetic mean over rolling time window')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return savefig('Override command rate')


def plot_override_comment_rate_multiple_jira_tickets(
        all_override_comments, ticketnames):

    # Create a new figure and an Axes object. Plot all line plots into the same
    # Axes object.
    fig = plt.figure()
    ax = fig.gca()

    for ticketname in ticketnames:
        ocs = [oc for oc in all_override_comments if oc['ticket'] == ticketname]
        commentrate_1, window_width_days_1, commentrate_2, window_width_days_2, _, _  = \
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
    set_title('Override command rate for most frequently tagged JIRA tickets')
    set_subtitle('Arithmetic mean over rolling time window')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
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


def analyze_merged_prs(prs, report):

    log.info('Filter merged pull requests.')
    filtered_prs = [pr for pr in prs if pr.merged_at is not None]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    # Hopefully helpful note: `pr.head.label` is e.g.
    # `mesosphere:mergebot/dcos/1.11/3292` for a pull request created in
    # mesosphere/dcos-enterprise created by Mergebot as of a bump-ee command.

    # Proceed with analyzing only those pull requests that were not created by
    # mergebot. This ignores an important class of pull request: all downstream
    # PRs created via the bump-ee command. This is an intentional, following the
    # idea that a pull request pair comprised of an upstream PR with the
    # corresponding downstream PR is tracking one unit of change to DC/OS.
    log.info('Filter pull requests not created by mergebot.')
    filtered_prs = [pr for pr in filtered_prs if 'mergebot' not in pr.user.login]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    log.info('Filter pull requests not having `Train` or `train` in title.')
    filtered_prs = [pr for pr in filtered_prs if 'train' not in pr.title.lower()]
    log.info('Number of filtered pull requests: %s', len(filtered_prs))

    # Major goal is to look at train
    # PRs separately, and to filter PRs by certain criteria in general, such as
    # - how many lines do they change
    # - are these just simple package bumps?
    # - ...

    # This line assumes that somewhere in the code path a figure has been
    # created before, now create a fresh one.
    plt.figure()

    log.info('Build main Dataframe')

    # I think the point in time when a pull request has been merged is the
    # better reference for determining metrics like throughput and latency than
    # the point in time when a pull request has been created.

    df = pd.DataFrame(
        {
            'created_at': [pr.created_at for pr in filtered_prs],
            'openseconds': [
                (pr.merged_at - pr.created_at).total_seconds() for
                pr in filtered_prs
            ]
        },
        index=[pd.Timestamp(pr.merged_at) for pr in filtered_prs]
        )

    # Sort by time.
    df.sort_index(inplace=True)

    df['opendays'] = df['openseconds'] / 86400

    latency, figure_latency_filepath = plot_latency(df)

    plt.figure()

    throughput, figure_throughput_filepath = plot_throughput(filtered_prs)

    plt.figure()

    quality = throughput / latency
    df['quality'] = quality

    figure_quality_filepath = plot_quality(df)

    #plt.show()

    report.write('\n\n## Pull request integration velocity\n\n')
    include_figure(
        report,
        figure_latency_filepath,
        'Pull request integration latency'
    )
    include_figure(
        report,
        figure_throughput_filepath,
        'Pull request integration throughput'
    )
    include_figure(
        report,
        figure_quality_filepath,
        'Pull request integration verlocity'
    )


def include_figure(report, filepath, heading):
    report.write(f'\n\n[![{heading}]({filepath} "{heading}")]({filepath})\n\n')


# What is good is low time to merge, and many pull requests merged per time.
# So, divide throughput by latency.


def plot_quality(df):
    df['quality'].plot()
    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day] / latency [day]')
    set_title('PR integration velocity for PRs in both DC/OS repos')
    # subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #     matcher.subtitle
    # set_subtitle('Raw data')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return savefig('Pull request integration velocity')


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
    # stddev = rollingwindow.std()

    throughput.plot(
        linestyle='dashdot',
        # linestyle='None',
        # marker='.',
        color='black',
        markersize=5,
    )
    plt.xlabel('Time')
    plt.ylabel('Throughput [1/day], rolling window of 3 weeks width')
    set_title('Pull request throughput for PRs in both DC/OS repos')
    # subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #     matcher.subtitle
    # set_subtitle('Raw data')
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    return throughput, savefig('Pull request integration throughput')


def plot_latency(df):

    df['opendays'].plot(
        # linestyle='dashdot',
        linestyle='None',
        color='gray',
        marker='.',
        markersize=3,
        markeredgecolor='gray'
    )
    plt.xlabel('Pull request creation time')
    plt.ylabel('Time-to-merge latency [day], rolling window of 3 weeks width')
    set_title('Time-to-merge for PRs in both DC/OS repositories')
    # subtitle = 'Freq spec from narrow rolling request rate -- ' + \
    #    matcher.subtitle
    set_subtitle('Raw data')
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    rollingwindow = df['opendays'].rolling('21d')
    mean = rollingwindow.mean()

    mean.plot(
        linestyle='solid',
        # linestyle='None',
        color='black',
        # marker='.',
        # markersize=1,
        # markeredgecolor='gray'
    )

    # stddev = rollingwindow.std()

    # plt.figure()

    # Rolling window of one week width,

    # rollingwindow = df['opendays'].rolling('7d')
    # mean = rollingwindow.mean()
    # stddev = rollingwindow.std()

    # mean.plot()

    return mean, savefig('Pull request intgration latency')


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
