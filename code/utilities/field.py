DESC = 'description'
SUMM = 'summary'
ISSUE_KEY = "issue_key"
PROJECT = "project"
ISSUE_TYPE = "issuetype"
TEXT = 'text'  # summ + desc
VOTES = 'votes'
WATCHERS = 'watchers'
PRIORITY = "priority"
WORDCOUNT = "wordcount"
REPORTER_ISSUENUM = "reporter_issuenum"
REPORTER_RECENT_ISSUENUM = "reporter_recent_issuenum"
REPORTER_STABLE_INFO_RATE = "reporter_stable_infochg_rate"
REPORTER_STABLE_SPRINT_RATE = "reporter_stable_sprint_rate"
ASSIGNEE_ISSUENUM = "assignee_issuenum"
ASSIGNEE_RECENT_ISSUENUM = "assignee_recent_issuenum"
ASSIGNEE_STABLE_INFO_RATE = "assignee_stable_infochg_rate"
ASSIGNEE_STABLE_SPRINT_RATE = "assignee_stable_sprint_rate"
COUNT_INFOCHG_BEFORE = 'count_infochg_before'
COUNT_COMMENT_BEFORE = 'count_comment_before'
ASSIGNEE_CHANGE_COUNT = 'assignee_change_count'


COLLAB_TOTAL_DEGREE = "collab_total_degree"
COLLAB_IN_DEGREE = "collab_in_degree"
COLLAB_OUT_DEGREE = "collab_out_degree"
COLLAB_KCORE = "collab_kcore"
COLLAB_CLUSTERING_COEFF = "collab_clustering_coeff"
COLLAB_CLOSENESS = "collab_closeness"
COLLAB_BETWEENNESS = "collab_betweenness"
COLLAB_EIGEN_VECTOR = "collab_eigen_vector"

Y_COSINE = 'y_cosine'

LABEL = "label"
FOLD = "fold"
RANDOM_ROUND = "randomRound"
CV_ROUND = "cvRound"

HAS_SUBTASKS = "has_subtasks"
HAS_EPIC = "has_epic"

def newFeatureList():
    return [
    COUNT_INFOCHG_BEFORE,
    COUNT_COMMENT_BEFORE,
    WORDCOUNT,
    ASSIGNEE_CHANGE_COUNT,
    'has_STACK',
    'has_CODE',
    'has_TESTCASE',
    'has_ATTACHMENT',
    'has_AC',
    'TASK',
    'LINK',
    COLLAB_TOTAL_DEGREE,
    COLLAB_IN_DEGREE,
    COLLAB_OUT_DEGREE,
    COLLAB_KCORE,
    COLLAB_CLUSTERING_COEFF,
    COLLAB_CLOSENESS,
    COLLAB_BETWEENNESS,
    COLLAB_EIGEN_VECTOR,
    REPORTER_ISSUENUM,
    REPORTER_RECENT_ISSUENUM,
    REPORTER_STABLE_INFO_RATE,
    REPORTER_STABLE_SPRINT_RATE,
    ASSIGNEE_ISSUENUM,
    ASSIGNEE_RECENT_ISSUENUM,
    ASSIGNEE_STABLE_INFO_RATE,
    ASSIGNEE_STABLE_SPRINT_RATE,
    VOTES,
    WATCHERS,
    PRIORITY,
    ISSUE_TYPE,
    "flesch",
    "fog",
    "lix",
    "kinkaid",
    "ari",
    "colemanlieu",
    "smog",
    HAS_EPIC,
    HAS_SUBTASKS
]

