## step 1: Ingestion
Set env vars
```
export LANCEDB_URI=<your-lancedb-uri>
export LANCEDB_API_KEY=<your-lancedb-api-key>
```

to ingest the dataset run this command:
```
modal run --detach ingest.py
```
By default it'll use table name 'wikipedia' and ingest entire 40M rows. You can change it by passing dataset fraction and table name like this:

```
modal run --detach ingest.py --down-scale 0.01 --table-name wiki
```

### Ingestion speed.
Depending on max containers and GPU concurrency, you'll see different ingestion speeds. LanceDB support high throughput rates of upto 4GB/sec. **In our test with max gpu concurrency 50GPUs, we were able to ingest the entire 41M rows in ~10mins.** The main bottleneck was GPU provision time. The ingestion speed can be further improved by keeping GPU instances warmed up to reduce provisioning time.


## Step 2: Query Endpoint
Deploy modal endpoint. ALong with `LANCEDB_URI` and `LANCEDB_API_KEY` it'll also need `LANCEDB_TABLE_NAME`
```
export LANCEDB_TABLE_NAME=<your-table-name>
```
Now deploy endpoint
```
modal deploy wikipedia-ingest.py
```

## step 3: frontend
* update the `frontend/src/config/endpoints.ts` file with your endpoint urls from previous section output.
* Run locally using `npm run dev`
* Deploy using `vercel`
