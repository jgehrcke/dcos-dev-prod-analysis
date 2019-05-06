import logging
import pickle
import sys


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(threadName)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S"
    )


class StrippedPullRequest:
    # No usage of __slots__ because the analyzer program stores more attributes
    # on these objects on the fly.
    pass


class StrippedIssueComment:
    __slots__ = ('created_at', 'body', 'user')


class StrippedIssueEvent:
    __slots__ = ('created_at', 'event', '_lowerlabelname')


PR_ATTRS = (
    'created_at',
    'merged_at',
    'title',
    'number',
    '_issue_comments',
    '_events',
    'user',  # user.login
)


COMMENT_ATTRS = (
    'created_at',
    'body',
    'user',
)


EVENT_ATTRS = (
    'event',
    'created_at',
    '_rawData',
    )


def main():

    infilepath = sys.argv[1]

    log.info('Unpickle from file: %s', infilepath)
    with open(infilepath, 'rb') as f:
        original_prs = pickle.load(f)

    stripped_prs = {}

    log.info('Create stripped-down object copies')
    for prnumber, opr in original_prs.items():
        o = StrippedPullRequest()
        for attr in PR_ATTRS:
            if attr == '_issue_comments':
                value_to_copy = strip_issue_comments(opr._issue_comments)
            if attr == '_events':
                value_to_copy = strip_issue_events(opr._events)
            else:
                value_to_copy = getattr(opr, attr)
            setattr(o, attr, value_to_copy)
        stripped_prs[prnumber] = o

    with open(infilepath + '.stripped', 'wb') as f:
        pickle.dump(stripped_prs, f)


def strip_issue_comments(original_ics):
    stripped_ics = []
    for oic in original_ics:
        c = StrippedIssueComment()
        for attr in COMMENT_ATTRS:
            setattr(c, attr, getattr(oic, attr))
        stripped_ics.append(c)
    return stripped_ics


def strip_issue_events(original_events):
    stripped_events = []
    for oe in original_events:
        if oe.event != 'labeled':
            # Do not keep these events around.
            continue
        e = StrippedIssueEvent()
        for attr in EVENT_ATTRS:
            if attr == '_rawData':
                value_to_copy = oe._rawData['label']['name'].lower()
                # In the analysis program this is all we need for now.
                attrname = '_lowerlabelname'
            else:
                attrname = attr
                value_to_copy = getattr(oe, attr)
            setattr(e, attrname, value_to_copy)
        stripped_events.append(e)
    return stripped_events


if __name__ == "__main__":
    main()
