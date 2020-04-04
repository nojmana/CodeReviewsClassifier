import sys
import pandas

from GitHubAPI.github_api import GitHubApi


def write_to_csv(data, user, project):
    filename = 'data_' + user + '_' + project + '.csv'

    df = pandas.DataFrame(data)
    df.to_csv(filename, index=False, header=True)


def read_token():
    file = open('token.txt', 'r')
    return file.read().split()[0]


if __name__ == "__main__":
    user = sys.argv[1]
    project = sys.argv[2]
    number_of_results = int(sys.argv[3])

    auth_token = read_token()

    github_api = GitHubApi(user, project, number_of_results, auth_token)
    comments = github_api.get_pull_request_comments_all_pages()
    write_to_csv(comments, user, project)
