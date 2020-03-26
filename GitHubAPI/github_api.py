import json
import urllib.request


class GitHubApi:

    def __init__(self, user, project, number_of_results):
        self.user = user
        self.project = project
        self.number_of_results = number_of_results

    def get_pull_request_comments_one_page(self, page_number):
        url = 'https://api.github.com/repos/' + self.user + '/' + self.project + \
              '/pulls/comments?per_page=100&page=' + str(page_number)
        with urllib.request.urlopen(url) as requested_url:
            json_data = json.loads(requested_url.read().decode())
            page_comments = list()
            for comment_data in json_data:
                page_comments.append(comment_data['body'])
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
