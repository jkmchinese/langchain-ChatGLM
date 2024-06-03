from .crawl_core import crawlCore

token = "NKd-kzFHcix232CDmZr3"
page = 0
issue_note_iid = 0
headers = {"PRIVATE-TOKEN": token}
# updatedAfterTime = '2024-05-26'

project_eds_id = 640
project_eds_name = "eds"

project_kp_id = 125
project_kp_name = "kp"

project_vmp_id = 503
project_vmp_name = "vmp"


def startCrawl(updatedAfterTime: str) -> int:
    count = 0
    print(f"eds start...")
    count += crawlCore(project_id=project_eds_id,
                       updatedAfterTime=updatedAfterTime,
                       token=token,
                       project_name=project_eds_name)
    print(f"eds end...")

    print(f"kp start...")
    count += crawlCore(project_id=project_kp_id,
                       updatedAfterTime=updatedAfterTime,
                       token=token,
                       project_name=project_kp_name)
    print(f"kp end...")

    print(f"vmp start...")
    count += crawlCore(project_id=project_vmp_id,
                       updatedAfterTime=updatedAfterTime,
                       token=token,
                       project_name=project_vmp_name)
    print(f"vmp end...")
