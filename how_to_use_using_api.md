###  Using the API with cURL

Once the server is running, you can easily test the endpoint from your terminal using `cURL`.

**Replace the placeholders** (`YOUR_BEARER_TOKEN`, the document URL, and your questions) in the command below:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
  "documents": "https://www.un.org/sites/un2.un.org/files/2021/08/sg_our_common_agenda_report_english.pdf",
  "questions": [
    "What is the main proposal of this report?",
    "How many key areas of action are there?"
  ]
}'
```
