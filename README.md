# DC/OS developer productivity analysis tools

## Usage

### Fetch data from GitHub

Set up a GitHub API token (`read` access to the relevant repositories is
sufficient).

Expose credentials to the program `dcos-dev-prod-fetchdata.py` via environment:

```
$ export GITHUB_APITOKEN="acf...663"
$ export GITHUB_USERNAME="username"
```

Fetch the relevant data from GitHub via the following two commands:
```
$ python dcos-dev-prod-fetchdata.py dcos/dcos
[...]

$python dcos-dev-prod-fetchdata.py mesosphere/dcos-enterprise
[...]
```

#### First invocation: fetch all data (slow, affected by GitHub API usage quota)

If the above two commands are invoked for the first time they can individually
take a long while to complete (like 10 minutes, depending on the connectivity to
GitHub). The output of the commands continuously informs about the number of
remaining HTTP requests before GitHub's hourly HTTP request quota hits in
(GitHub allows 5.000 API HTTP requests per hour).

The program is not yet built for the case where the hourly quota is exhausted
within a single invocation.

At the time of writing the first command consumes a significant fraction of the
hourly quota (roughly 4.000) so that the second command should only be executed
roughly one hour after the first command has succeeded.

#### Subsequent invocations: best-effort update (fast)

The program writes the data to CPython pickle files to the current working
directory and discovers those upon subsequent invocations. If a subsequent
invocation is performed just a small number of days after the last 'complete
fetch' then it performs a best-effort update. This best-effort update might miss
updates in really old pull requests (a 'complete fetch' should be performed
regularly so that the accumulated error does not get too big over time).


### Analyze data, render Markdown report as HTML

Invoke the analysis program and point it to a [pandoc](https://pandoc.org/)
executable:

```
$ python dcos-dev-prod-analysis.py --pandoc-command=./pandoc-2.2.3.2/bin/pandococ-2.2.3.2/bin/
181126-11:45:40.593 INFO: Unpickle from file: dcos-enterprise_pull-requests-with-comments.pickle
181126-11:45:44.178 INFO: Unpickle from file: dcos_pull-requests-with-comments.pickle
181126-11:45:49.314 INFO: Create output directory: 2018-11-26_report
181126-11:45:49.350 INFO: Perform comment analysis for 7673 PRs

[...]

181126-11:45:54.852 INFO: Copy resources directory into output directory
181126-11:45:54.854 INFO: Trying to run Pandoc for generating HTML document
181126-11:45:54.854 INFO: Running command: ./pandoc-2.2.3.2/bin/pandoc --toc --standalone --template=resources/template.html 2018-11-26_report/2018-11-26_dcos-dev-prod-report.md -o 2018-11-26_report/2018-11-26_dcos-dev-prod-report.html
181126-11:45:55.321 INFO: Pandoc terminated indicating success
```

Open the generated report HTML document in a browser.