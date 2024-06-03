import requests
import pandas as pd
import json
import re

# curl --location 'http://git.rdapp.com/api/v4/projects/125/issues?state=opened&iid=&labels=&milestone=&order_by=created_at&sort=desc&page=0&per_page=20&created_after=2023-06-02' \
# --header 'PRIVATE-TOKEN: NKd-kzFHcix232CDmZr3'

token = "NKd-kzFHcix232CDmZr3"
project_id = 503
page = 0
issue_note_iid = 0
project_name = "vmp"
headers = {"PRIVATE-TOKEN": token}


def convert(issue):
    issue_note_iid = issue["iid"]
    notes_url = f"http://git.rdapp.com/api/v4/projects/{project_id}/issues/{issue_note_iid}/notes?page=0&per_page=100"
    # data-constructure gitlab-issue.json
    notes_response = requests.get(notes_url, headers=headers).json()
    excluded_words = r'^(mentioned|assigned|changed|marked|reopened|created|closed)\b'
    notes = list(
        map(lambda n: f'{n["author"]["name"]}:{n["body"]}',
            filter(lambda n: not re.match(excluded_words, n["body"], re.IGNORECASE) and not n["author"]["name"] == 'administrator', notes_response))
    )

    # for n in notes:
    #     print(n)
    # data-constructure gitlab-issue.json
    new_issue = {
        "id": issue['iid'],
        "标题": issue['title'],
        "描述": issue['description'],
        "标签": issue['labels'],
        # message = "A" if age >=18 else "B"
        "回复": notes
    }
    return new_issue


def generate_lines(issue):
    yield f'## 标题\n\n{issue["标题"]}\n\n'
    yield f'## 描述\n\n{issue["描述"]}\n\n'
    biaoqian = ",".join(issue["标签"])
    yield f'\n\n## 标签\n\n{biaoqian}\n\n'
    dafu = "\n\n".join(issue["回复"])
    yield f'\n\n## 回复\n\n{dafu}\n\n'


while True:
    issue_url = f"http://git.rdapp.com/api/v4/projects/{project_id}/issues?state=all&iid=&labels=&milestone=&order_by=updated_at&sort=desc&page={page}&per_page=100&updated_after=2024-05-26'"
    headers = {"PRIVATE-TOKEN": token}
    print(f"issue_url:{issue_url}")
    response = requests.get(issue_url, headers=headers).json()
    issues = list(map(convert, response))
    print(f"issues count {len(issues)}")
    for issue in issues:
        # print(issue["标题"])
        with open(f'../content/gitlab/{project_name}/{project_name}-{issue["id"]}.md', 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(generate_lines(issue))

    if len(issues) == 0:
        print("issues count 0, exit...")
        break
    else:
        page += 1
