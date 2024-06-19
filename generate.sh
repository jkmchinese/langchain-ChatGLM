curl --location 'http://127.0.0.1:7861/generate' \
--header 'Content-Type: application/json' \
--data '{
    "knowledge_base_id": "vmp"
}'

curl --location 'http://127.0.0.1:7861/generate' \
--header 'Content-Type: application/json' \
--data '{
    "knowledge_base_id": "eds"
}'

curl --location 'http://127.0.0.1:7861/generate' \
--header 'Content-Type: application/json' \
--data '{
    "knowledge_base_id": "kp"
}'