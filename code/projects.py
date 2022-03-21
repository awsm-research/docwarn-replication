

class Project(object):
    key = ""
    abb = ""
    url = ""
    requireLogin = False
    githubProjectName = ""
    githubRepoName = ""

    def __str__(self):
        return self.key

    def __init__(self, key, abb, url, requireLogin, githubRepoName='', githubProjectName=''):
        self.key = key
        self.abb = abb
        self.url = url
        self.requireLogin = requireLogin
        self.githubRepoName = githubRepoName
        self.githubProjectName = githubProjectName

DM = Project('DM', 'DM', 'https://jira.lsstcorp.org', False, 'https://github.com/lsst-dm', '')
MESOS = Project('MESOS', 'ME', 'https://issues.apache.org/jira', True, '', 'https://github.com/apache/mesos')
TDQ = Project('TDQ', 'TD', 'https://jira.talendforge.org', False, '', 'https://github.com/Talend/data-quality')
TIMOB = Project('TIMOB', 'TI', 'https://jira.appcelerator.org', False, '','https://github.com/appcelerator/titanium_mobile')
TDP = Project('TDP', 'TDP', 'https://jira.talendforge.org', True)
TMDM = Project('TMDM', 'TMDM', 'https://jira.talendforge.org', True)
XD = Project('XD', 'XD', 'https://jira.spring.io', True, '', 'https://github.com/spring-projects/spring-xd')
TUP = Project('TUP', 'TUP', 'https://jira.talendforge.org', True)
DATALAB = Project('DATALAB', 'DATALAB', 'https://issues.apache.org/jira', True)

infoChgList = [DM, MESOS, TDP, TDQ, TIMOB, TMDM, TUP, DATALAB, XD]

