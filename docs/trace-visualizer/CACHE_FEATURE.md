# Cache Feature Documentation

## Overview

The trace visualizer now includes an **automatic caching system** that saves your annotation work to the browser's localStorage. This means you can refresh the page, close the browser, and come back later without losing your progress.

## How It Works

### Automatic Saving
- **Every annotation you make is automatically saved** to the browser's cache
- This includes:
  - Behavioral markers on nodes
  - Node observations/notes
  - Overall trace comments
  - Your progress across all files

### Cache Key
- Annotations are cached **per user** based on your:
  - Annotator Name
  - Personal Identifier
- This means different annotators can work on the same browser without conflicts

### When Cache is Loaded
1. **On page load** - if you have annotator credentials entered
2. **When loading files** - if the files match a cached session
3. **When entering credentials** - cache is loaded when you enter both name and identifier

### When Cache is Saved
- Every time you:
  - Add or remove a marker
  - Type notes or observations
  - Switch between files
  - Change trace comments

### When Cache is Cleared
- **Automatically** after successful submission to the API
- **Manually** when you click the "🗑️ Clear Cache" button
- **On cache version updates** (when the app is updated)

## Performance Overhead

**Minimal to negligible**:
- Uses browser's native `localStorage` API (extremely fast)
- Operations are synchronous and take < 1ms typically
- No network requests involved
- No server-side infrastructure needed
- Storage limit: ~5-10MB per domain (more than enough for annotations)

## Browser Compatibility

Works in all modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Privacy & Security

- All data stays in **your browser only**
- Nothing is sent to servers until you click "Submit Annotations"
- Cache is cleared after successful submission
- Other users on the same computer would need your exact credentials to see your cache

## Benefits

✅ **No lost work** - refresh the page anytime without worry
✅ **Resume anytime** - close the browser and continue later
✅ **Multi-session** - work on different batches at different times
✅ **Instant** - no loading delays
✅ **Offline-capable** - annotations work even without internet (submission requires connection)

## Technical Details

### Cache Structure
```javascript
{
  version: "1.0",
  timestamp: 1699123456789,
  annotations: { /* per-file, per-node annotations */ },
  nodes: { /* annotatable nodes info */ },
  traceComments: { /* per-file trace comments */ },
  currentFileIndex: 0,
  fileNames: ["file1.json", "file2.json"]
}
```

### Cache Key Format
```
trace_annotator_cache_1.0_<annotatorName>_<personalIdentifier>
```

### Storage Size
Typical annotation session: **50-200 KB**
Maximum practical limit: **5-10 MB** (thousands of files)

## Troubleshooting

### Cache not loading?
1. Make sure you entered the same Annotator Name and Personal Identifier
2. Check that you loaded the same set of files
3. Try opening browser console (F12) and look for cache-related messages

### Running out of space?
- Very unlikely, but if it happens:
  - Click "Clear Cache" to free up space
  - Submit annotations more frequently
  - Use browser's developer tools to manually clear localStorage

### Want to start fresh?
- Click the "🗑️ Clear Cache" button
- Or change your Annotator Name/Personal Identifier

## For Developers

### Key Functions
- `saveCachedData()` - Save current state to localStorage
- `loadCachedData()` - Restore state from localStorage
- `clearCachedData()` - Remove cached data
- `autoSaveAnnotations()` - Wrapper that saves current file + cache
- `getCacheKey()` - Generate unique cache key for session

### Integration Points
- Marker changes → auto-save
- Note edits → auto-save (debounced via input event)
- File switching → auto-save
- Folder load → load cache
- Successful submission → clear cache
