# UI Prototype

Open `index.html` in a browser, or serve this directory from the API gateway.

The UI renders the backend MJPEG stream from `/v1/video.mjpg` and listens to
`/v1/fusion/events` for real fusion events. It does not request browser camera
access and does not synthesize mock events.

Expected event shape:

```json
{
  "utterance_id": "utt_42",
  "text": "你好",
  "person_track_ids": [1],
  "position": "left",
  "confidence": 0.87,
  "tentative": false,
  "tracks": [
    {
      "person_track_id": 1,
      "bbox": {"x1": 120, "y1": 80, "x2": 420, "y2": 640},
      "quality": "good",
      "p_active": 0.84
    }
  ]
}
```
