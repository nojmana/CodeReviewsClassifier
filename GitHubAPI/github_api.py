import json
import urllib.request


def get_pull_request_comments(user, project):
    project_url = 'https://api.github.com/repos/' + user + '/' + project + '/pulls/comments'
    print(project_url)
    with urllib.request.urlopen(project_url) as url:
        json_data = json.loads(url.read().decode())
        comments = []
        for comment_data in json_data:
            comments.append(comment_data['body'])
        return comments


if __name__ == "__main__":
    user = 'eclipse'
    project = 'openj9'
    data = get_pull_request_comments(user, project)
    print(data)
