# CodeReviewsClassifier

## GitHubAPI
A script downloading code review comments from GitHubAPI.

#### Input
Command line arguments:
```
* GitHub username 
* GitHub repository name
* number of reviews to be downloaded

f.ex: eclipse openj9 10
```

Generate Personal access token for GitHubAPI and paste it to token.txt file in order to increase default rate limit.

#### Output
Csv file with each comment on a new line. An example file can be found in the repository (data_eclipse_openj9.csv). 
