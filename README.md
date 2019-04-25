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

If the above two commands are invoked for the first time they will individually
take a long while to complete (on the order of hours, even with good
connectivity to GitHub). This is because for each repository the fetcher program
needs to perform thousands of HTTP requests. The output of the fetcher program
continuously informs about the progress.

While interacting with GitHub the fetcher program is expected to run into quota
and rate limit errors emitted by the GitHub API. It handles those errors by
backing off, waiting, and retrying. The fetcher program also handles most
transport and HTTP errors by retrying. If it runs on a laptop it is okay to put
the laptop to sleep during the data collection process -- the collection is
expected to continue just fine upon resumption.


#### Subsequent invocations: best-effort update (faster)

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