import json
import urllib.request


class GitHubApi:

    def __init__(self, user, project, number_of_results, auth_token):
        self.user = user
        self.project = project
        self.number_of_results = number_of_results
        self.auth_token = auth_token

    def authorization_header(self):
        if self.auth_token == 'paste_here_your_personal_access_token':
            print('No authorization token provided.')
            return {}
        headers = {'Authorization': 'token %s' % self.auth_token}
        return headers

    def get_pull_request_author(self, comment):
        request = urllib.request.Request(comment['pull_request_url'], headers=self.authorization_header())
        with urllib.request.urlopen(request) as requested_url:
            json_data = json.loads(requested_url.read().decode())
            return json_data['user']['id']

    def check_if_reviewer_is_author(self, comment):
        comment_author = comment['user']['id']
        if comment_author == self.get_pull_request_author(comment):
            return True
        else:
            return False

    def get_pull_request_comments_one_page(self, page_number):
        per_page = 100
        url = 'https://api.github.com/repos/' + self.user + '/' + self.project + \
              '/pulls/comments?per_page=' + str(per_page) + '&page=' + str(page_number)
        request = urllib.request.Request(url, headers=self.authorization_header())
        with urllib.request.urlopen(request) as requested_url:
            json_data = json.loads(requested_url.read().decode())
            page_comments = list()
            for comment in json_data:
                print('result nr', page_number * per_page + len(page_comments) + 1, 'out of', self.number_of_results)
                page_comments.append({'FILENAME': comment['path'],
                                      'CR AUTHOR = PR AUTHOR': self.check_if_reviewer_is_author(comment),
                                      'BODY': comment['body']})
                if len(page_comments) >= self.number_of_results:
                    break
            return page_comments

    def get_pull_request_comments_all_pages(self):
        all_comments = list()
        page_number = 0
        number_of_results_counter = 0
        while number_of_results_counter < self.number_of_results:
            page_comments = self.get_pull_request_comments_one_page(page_number)
            if len(page_comments) == 0:
                break
            number_of_results_counter += len(page_comments)
            page_number += 1
            all_comments += page_comments
        return all_comments
