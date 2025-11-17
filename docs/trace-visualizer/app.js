let allFiles = [];
let currentFileIndex = 0;
let currentFilter = 'all';
let svg, g, simulation, zoomBehavior;
let currentNodes = []; // Store current graph nodes
let selectedNodeIndex = -1; // Track selected node index (-1 means none selected)

// API endpoint configuration
const API_ENDPOINT = 'https://lamalab-org--llm-annotation-endpoint-fastapi-app.modal.run/ingest';

// Storage for behavioral markers and notes per node, per file
let allFileAnnotations = {}; // Format: { fileName: { nodeId: { markers: [], notes: '' } } }
let allFileNodes = {}; // Format: { fileName: [nodes array] } - to store which nodes are annotatable
let nodeAnnotations = {}; // Current file's annotations

// Storage for trace-level comments per file
let allFileTraceComments = {}; // Format: { fileName: 'trace comment text' }

// Cache configuration
const CACHE_VERSION = '1.0'; // Increment this to invalidate old caches
const CACHE_KEY = 'trace_annotator_cache_v1'; // Single cache key for all sessions

// Check for cached session on startup and show notification
window.addEventListener('DOMContentLoaded', function() {
    checkForCachedSession();
});

// Color scheme
const colors = {
    system: '#6c757d',
    user: '#28a745',
    assistant: '#667eea',
    tool: '#fd7e14',
    result: '#17a2b8'
};

// Folder input handler
document.getElementById('folderInput').addEventListener('change', function(e) {
    const files = Array.from(e.target.files).filter(f => f.name.endsWith('.json'));

    if (files.length === 0) {
        alert('No JSON files found in the selected directory');
        return;
    }

    // Sort files by name
    files.sort((a, b) => a.name.localeCompare(b.name));
    allFiles = files;
    currentFileIndex = 0;

    // Populate file selector
    const select = document.getElementById('fileSelect');
    select.innerHTML = '';
    files.forEach((file, idx) => {
        const option = document.createElement('option');
        option.value = idx;
        option.textContent = file.name;
        select.appendChild(option);
    });

    // Show file selector
    document.getElementById('fileSelectorGroup').style.display = 'flex';
    document.getElementById('totalFiles').textContent = files.length;

    // Try to load cached data for these files
    loadCachedData();

    // Load first file
    loadFileByIndex(0);
});

// File select dropdown handler
document.getElementById('fileSelect').addEventListener('change', function(e) {
    loadFileByIndex(parseInt(e.target.value));
});

// Navigation button handlers
document.getElementById('prevBtn').addEventListener('click', function() {
    if (currentFileIndex > 0) {
        loadFileByIndex(currentFileIndex - 1);
    }
});

document.getElementById('nextBtn').addEventListener('click', function() {
    if (currentFileIndex < allFiles.length - 1) {
        loadFileByIndex(currentFileIndex + 1);
    }
});

// Filter button handlers
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentFilter = this.dataset.filter;
        if (allFiles.length > 0) {
            loadFileByIndex(currentFileIndex);
        }
    });
});

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (allFiles.length === 0) return;

    if (e.key === 'ArrowLeft' && currentFileIndex > 0) {
        loadFileByIndex(currentFileIndex - 1);
    } else if (e.key === 'ArrowRight' && currentFileIndex < allFiles.length - 1) {
        loadFileByIndex(currentFileIndex + 1);
    }
});

// Help toggle button handler
document.getElementById('helpToggleBtn').addEventListener('click', function() {
    const helpDescriptions = document.getElementById('helpDescriptions');
    const isVisible = helpDescriptions.style.display !== 'none';

    if (isVisible) {
        helpDescriptions.style.display = 'none';
        this.textContent = 'ℹ️ Show Descriptions';
    } else {
        helpDescriptions.style.display = 'block';
        this.textContent = 'ℹ️ Hide Descriptions';
    }
});

// Clear Cache button handler
document.getElementById('clearCacheBtn').addEventListener('click', function() {
    if (confirm('Are you sure you want to clear all cached annotations? This cannot be undone.')) {
        clearCachedData();

        // Reset current annotations
        allFileAnnotations = {};
        allFileNodes = {};
        allFileTraceComments = {};
        nodeAnnotations = {};

        // Clear UI
        document.getElementById('traceCommentsTextarea').value = '';
        document.getElementById('notesTextarea').value = '';
        document.getElementById('selectedMarkers').innerHTML = '';

        // Reload current file to reset visualization
        if (allFiles.length > 0) {
            loadFileByIndex(currentFileIndex);
        }
    }
});

// Marker dropdown handlers
document.getElementById('neutralMarkerSelect').addEventListener('change', function(e) {
    if (selectedNodeIndex === -1 || this.disabled || !e.target.value) return;
    addMarkerFromDropdown(e.target.value);
    this.value = ''; // Reset dropdown
});

document.getElementById('positiveMarkerSelect').addEventListener('change', function(e) {
    if (selectedNodeIndex === -1 || this.disabled || !e.target.value) return;
    addMarkerFromDropdown(e.target.value);
    this.value = ''; // Reset dropdown
});

document.getElementById('negativeMarkerSelect').addEventListener('change', function(e) {
    if (selectedNodeIndex === -1 || this.disabled || !e.target.value) return;
    addMarkerFromDropdown(e.target.value);
    this.value = ''; // Reset dropdown
});

function addMarkerFromDropdown(marker) {
    const nodeId = currentNodes[selectedNodeIndex].id;

    // Initialize annotations for this node if not exists
    if (!nodeAnnotations[nodeId]) {
        nodeAnnotations[nodeId] = { markers: [], notes: '' };
    }

    // Only add if not already present
    if (!nodeAnnotations[nodeId].markers.includes(marker)) {
        nodeAnnotations[nodeId].markers.push(marker);
        updateMarkersDisplay(nodeId);

        // Refresh node colors to reflect which nodes can now be annotated
        updateNodeAnnotatability();

        // Auto-save annotations to cache
        autoSaveAnnotations();
    }
}

// Annotator info change handlers - reload cache when identifiers are entered
document.getElementById('annotatorName').addEventListener('blur', function() {
    if (this.value.trim() && document.getElementById('mongodbKey').value.trim()) {
        loadCachedData();
    }
});

document.getElementById('mongodbKey').addEventListener('blur', function() {
    if (this.value.trim() && document.getElementById('annotatorName').value.trim()) {
        loadCachedData();
    }
});

// Clear cache button handler
document.getElementById('clearCacheBtn').addEventListener('click', function() {
    if (confirm('Are you sure you want to clear all cached annotations? This action cannot be undone.')) {
        clearCachedData();

        // Reset current session data
        allFileAnnotations = {};
        allFileTraceComments = {};
        nodeAnnotations = {};

        // Clear UI
        document.getElementById('traceCommentsTextarea').value = '';
        document.getElementById('notesTextarea').value = '';
        document.getElementById('selectedMarkers').innerHTML = '';

        // Reload current file to refresh display
        if (allFiles.length > 0) {
            loadFileByIndex(currentFileIndex);
        }
    }
});

// Marker button handlers - Set up event delegation for marker buttons (keeping for backwards compatibility)
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('marker-btn')) {
        if (selectedNodeIndex === -1 || e.target.disabled) return;

        const nodeId = currentNodes[selectedNodeIndex].id;
        const marker = e.target.dataset.marker;

        // Initialize annotations for this node if not exists
        if (!nodeAnnotations[nodeId]) {
            nodeAnnotations[nodeId] = { markers: [], notes: '' };
        }

        // Toggle marker - if already present, remove it; otherwise add it
        const markerIndex = nodeAnnotations[nodeId].markers.indexOf(marker);
        if (markerIndex > -1) {
            nodeAnnotations[nodeId].markers.splice(markerIndex, 1);
            e.target.classList.remove('selected');
        } else {
            nodeAnnotations[nodeId].markers.push(marker);
            e.target.classList.add('selected');
        }

        updateMarkersDisplay(nodeId);

        // Refresh node colors to reflect which nodes can now be annotated
        updateNodeAnnotatability();

        // Auto-save annotations to cache
        autoSaveAnnotations();
    }
});

// Trace comments textarea handler
document.getElementById('traceCommentsTextarea').addEventListener('input', function() {
    // Save trace comments for current file
    if (allFiles.length > 0 && allFiles[currentFileIndex]) {
        const currentFileName = allFiles[currentFileIndex].name;
        allFileTraceComments[currentFileName] = this.value;

        // Auto-save annotations to cache
        autoSaveAnnotations();
    }
});

// Notes textarea handler
document.getElementById('notesTextarea').addEventListener('input', function() {
    if (selectedNodeIndex === -1) return;

    const nodeId = currentNodes[selectedNodeIndex].id;

    // Initialize annotations for this node if not exists
    if (!nodeAnnotations[nodeId]) {
        nodeAnnotations[nodeId] = { markers: [], notes: '' };
    }

    // Save notes
    nodeAnnotations[nodeId].notes = this.value;

    // Refresh node colors to reflect which nodes can now be annotated
    updateNodeAnnotatability();

    // Auto-save annotations to cache
    autoSaveAnnotations();
});

function updateMarkersDisplay(nodeId) {
    const container = document.getElementById('selectedMarkers');
    const annotations = nodeAnnotations[nodeId];

    if (!annotations || annotations.markers.length === 0) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    annotations.markers.forEach(marker => {
        html += `
            <div class="marker-tag">
                <span>${marker}</span>
                <span class="marker-tag-remove" data-marker="${marker}" data-node="${nodeId}">×</span>
            </div>
        `;
    });

    container.innerHTML = html;

    // Add remove handlers
    container.querySelectorAll('.marker-tag-remove').forEach(btn => {
        btn.addEventListener('click', function() {
            const marker = this.dataset.marker;
            const nodeId = parseInt(this.dataset.node);
            removeMarker(nodeId, marker);
        });
    });
}

function removeMarker(nodeId, marker) {
    if (!nodeAnnotations[nodeId]) return;

    const index = nodeAnnotations[nodeId].markers.indexOf(marker);
    if (index > -1) {
        nodeAnnotations[nodeId].markers.splice(index, 1);
        updateMarkersDisplay(nodeId);

        // Refresh node colors to reflect which nodes can now be annotated
        updateNodeAnnotatability();
    }
}

function loadNodeAnnotations(nodeId) {
    const annotations = nodeAnnotations[nodeId];

    // Update markers display
    updateMarkersDisplay(nodeId);

    // Update marker button states (for backwards compatibility if buttons are used)
    document.querySelectorAll('.marker-btn').forEach(btn => {
        const marker = btn.dataset.marker;
        if (annotations && annotations.markers.includes(marker)) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });

    // Update notes textarea
    const notesTextarea = document.getElementById('notesTextarea');
    notesTextarea.value = annotations ? annotations.notes : '';
}

function isNodeAnnotatable(node) {
    // Don't allow annotations for system nodes
    if (node.type === 'system') {
        return false;
    }

    // Don't allow annotations for tool nodes
    if (node.type === 'tool') {
        return false;
    }

    // Don't allow annotations for the first user node
    if (node.type === 'user') {
        // Check if this is the first user node in currentNodes
        const firstUserNode = currentNodes.find(n => n.type === 'user');
        if (firstUserNode && firstUserNode.id === node.id) {
            return false;
        }
    }

    // Check if all previous annotatable nodes have been labeled
    if (!allPreviousNodesLabeled(node)) {
        return false;
    }

    return true;
}

function getAnnotationDisabledReason(node) {
    // Provide specific reason why a node cannot be annotated
    if (node.type === 'system') {
        return 'System nodes cannot be annotated.';
    }

    if (node.type === 'tool') {
        return 'Tool nodes cannot be annotated.';
    }

    if (node.type === 'user') {
        const firstUserNode = currentNodes.find(n => n.type === 'user');
        if (firstUserNode && firstUserNode.id === node.id) {
            return 'The first user node cannot be annotated.';
        }
    }

    if (!allPreviousNodesLabeled(node)) {
        return 'Please label all previous annotatable nodes first before labeling this one.';
    }

    return 'This node cannot be annotated.';
}

function allPreviousNodesLabeled(node) {
    // Find the index of the current node
    const nodeIndex = currentNodes.findIndex(n => n.id === node.id);
    if (nodeIndex === -1) return true;

    // Check all previous nodes
    for (let i = 0; i < nodeIndex; i++) {
        const prevNode = currentNodes[i];

        // Skip non-annotatable node types (system, tool, first user)
        if (prevNode.type === 'system' || prevNode.type === 'tool') {
            continue;
        }

        // Check if this is the first user node (which is not annotatable)
        if (prevNode.type === 'user') {
            const firstUserNode = currentNodes.find(n => n.type === 'user');
            if (firstUserNode && firstUserNode.id === prevNode.id) {
                continue;
            }
        }

        // This is an annotatable node - check if it has been labeled
        const hasAnnotations = nodeAnnotations[prevNode.id] &&
                              (nodeAnnotations[prevNode.id].markers.length > 0 ||
                               nodeAnnotations[prevNode.id].notes.trim().length > 0);

        if (!hasAnnotations) {
            return false; // Found an unlabeled previous annotatable node
        }
    }

    return true; // All previous annotatable nodes are labeled
}

// Node navigation button handlers
document.getElementById('prevNodeBtn').addEventListener('click', function() {
    if (selectedNodeIndex > 0) {
        navigateToNode(selectedNodeIndex - 1);
    }
});

document.getElementById('nextNodeBtn').addEventListener('click', function() {
    if (selectedNodeIndex < currentNodes.length - 1) {
        if (selectedNodeIndex === -1) {
            // First click - go to first node
            navigateToNode(0);
        } else {
            navigateToNode(selectedNodeIndex + 1);
        }
    }
});

function navigateToNode(index) {
    if (index < 0 || index >= currentNodes.length) return;

    selectedNodeIndex = index;
    const node = currentNodes[index];

    // Highlight the selected node and show its details
    showDetails(null, node);

    // Update the visual selection in the graph
    d3.selectAll('.node circle')
        .attr('stroke-width', d => d.id === node.id ? 4 : 1)
        .attr('stroke', d => d.id === node.id ? '#ff6b6b' : '#333');

    // Center the node in the viewport with smooth animation
    if (svg && zoomBehavior) {
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Calculate the transform needed to center the node
        const scale = 1.5; // Zoom level for focused view
        const x = width / 2 - node.x * scale;
        const y = height / 2 - node.y * scale;

        // Animate the zoom/pan to center the node
        svg.transition()
            .duration(750)
            .call(
                zoomBehavior.transform,
                d3.zoomIdentity.translate(x, y).scale(scale)
            );
    }

    // Update button states
    updateNodeNavButtons();
}

function updateNodeNavButtons() {
    const prevBtn = document.getElementById('prevNodeBtn');
    const nextBtn = document.getElementById('nextNodeBtn');

    prevBtn.disabled = selectedNodeIndex <= 0;
    nextBtn.disabled = selectedNodeIndex >= currentNodes.length - 1;
}

function loadFileByIndex(index) {
    if (index < 0 || index >= allFiles.length) return;

    // Save current file's annotations and trace comments before switching
    if (allFiles.length > 0 && allFiles[currentFileIndex]) {
        const currentFileName = allFiles[currentFileIndex].name;
        allFileAnnotations[currentFileName] = JSON.parse(JSON.stringify(nodeAnnotations));
        // Save trace comments for current file
        const traceCommentsTextarea = document.getElementById('traceCommentsTextarea');
        allFileTraceComments[currentFileName] = traceCommentsTextarea.value;

        // Auto-save to cache when switching files
        autoSaveAnnotations();
    }

    currentFileIndex = index;
    const file = allFiles[index];

    const reader = new FileReader();
    reader.onload = function(event) {
        try {
            const data = JSON.parse(event.target.result);
            visualizeTrace(data);

            // Update UI
            document.getElementById('fileSelect').value = index;
            document.getElementById('currentFile').textContent = index + 1;
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === allFiles.length - 1;
        } catch (error) {
            alert('Error parsing ' + file.name + ': ' + error.message);
        }
    };
    reader.readAsText(file);
}

function clearDetailsPanel() {
    const content = document.getElementById('detailsContent');
    content.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">Select a node to view details</p>';

    // Clear annotations display
    document.getElementById('selectedMarkers').innerHTML = '';
    document.getElementById('notesTextarea').value = '';

    // Disable marker dropdowns
    document.getElementById('neutralMarkerSelect').disabled = true;
    document.getElementById('positiveMarkerSelect').disabled = true;
    document.getElementById('negativeMarkerSelect').disabled = true;

    // Reset dropdown values
    document.getElementById('neutralMarkerSelect').value = '';
    document.getElementById('positiveMarkerSelect').value = '';
    document.getElementById('negativeMarkerSelect').value = '';

    // Disable marker buttons (for backwards compatibility) and notes textarea when no node is selected
    document.querySelectorAll('.marker-btn').forEach(btn => {
        btn.disabled = true;
        btn.classList.remove('selected');
    });
    document.getElementById('notesTextarea').disabled = true;

    // Hide disabled message
    const disabledMsg = document.getElementById('annotationDisabledMsg');
    if (disabledMsg) {
        disabledMsg.style.display = 'none';
    }

    // Clear visual selection
    d3.selectAll('.node circle')
        .attr('stroke-width', 1)
        .attr('stroke', '#333');
}

function visualizeTrace(data) {
    const agentType = data.agent || 'ToolCallingAgent'; // Default to ToolCallingAgent
    const messages = data.messages;
    const nodes = [];
    const links = [];
    let nodeId = 0;
    let lastNodeId = -1;

    // Reset node selection when loading new trace
    selectedNodeIndex = -1;

    // Load annotations for the new file
    const newFileName = allFiles[currentFileIndex]?.name;
    if (newFileName && allFileAnnotations[newFileName]) {
        nodeAnnotations = JSON.parse(JSON.stringify(allFileAnnotations[newFileName]));
    } else {
        nodeAnnotations = {};
    }

    // Load trace comments for the new file
    const traceCommentsTextarea = document.getElementById('traceCommentsTextarea');
    if (newFileName && allFileTraceComments[newFileName]) {
        traceCommentsTextarea.value = allFileTraceComments[newFileName];
    } else {
        traceCommentsTextarea.value = '';
    }

    // Clear the details panel
    clearDetailsPanel();

    // Update agent type display
    document.getElementById('agentType').textContent = agentType;

    // Check agent type and process accordingly
    if (agentType === 'ToolCallingAgent') {
        visualizeToolCallingAgent(messages, nodes, links, nodeId, lastNodeId);
    } else if (agentType === 'ReActAgent') {
        visualizeReActAgent(messages, nodes, links, nodeId, lastNodeId);
    } else {
        // Fallback to ToolCallingAgent behavior
        visualizeToolCallingAgent(messages, nodes, links, nodeId, lastNodeId);
    }

    // Update stats
    document.getElementById('totalNodes').textContent = nodes.length;
    document.getElementById('toolCalls').textContent =
        nodes.filter(n => n.type === 'tool').length;

    // Store current nodes for navigation
    currentNodes = nodes;

    // Store nodes for this file for validation purposes
    const currentFileName = allFiles[currentFileIndex]?.name;
    if (currentFileName) {
        allFileNodes[currentFileName] = nodes.map(node => ({
            id: node.id,
            type: node.type,
            annotatable: isNodeAnnotatable(node)
        }));
    }

    drawGraph(nodes, links);

    // Update node navigation buttons
    updateNodeNavButtons();
}

function visualizeToolCallingAgent(messages, nodes, links, nodeId, lastNodeId) {
    // Process messages
    messages.forEach((msg, idx) => {
        const role = msg.role;

        if (currentFilter === 'all') {
            // ALL MODE: Show all messages including tool results
            nodes.push({
                id: nodeId,
                type: role,
                content: msg.content || '',
                timestamp: msg.timestamp || '',
                index: idx,
                toolCallId: msg.tool_call_id,
                toolName: msg.name,
                toolCalls: msg.tool_calls || null  // Add tool_calls to node
            });

            if (nodeId > 0) {
                links.push({
                    source: nodeId - 1,
                    target: nodeId
                });
            }
            nodeId++;
        } else if (currentFilter === 'tools') {
            // TOOLS ONLY MODE: Show only tool result messages (role='tool')
            if (role === 'tool') {
                nodes.push({
                    id: nodeId,
                    type: 'tool',
                    content: msg.content || '',
                    timestamp: msg.timestamp || '',
                    index: idx,
                    toolCallId: msg.tool_call_id,
                    toolName: msg.name,
                    toolCalls: msg.tool_calls || null
                });

                // Connect to previous tool node
                if (lastNodeId >= 0) {
                    links.push({
                        source: lastNodeId,
                        target: nodeId
                    });
                }
                lastNodeId = nodeId;
                nodeId++;
            }
        } else if (currentFilter === 'no-tools') {
            // NO TOOLS MODE: Show all messages except tool results
            if (role !== 'tool') {
                nodes.push({
                    id: nodeId,
                    type: role,
                    content: msg.content || '',
                    timestamp: msg.timestamp || '',
                    index: idx,
                    toolCallId: msg.tool_call_id,
                    toolName: msg.name,
                    toolCalls: msg.tool_calls || null
                });

                if (lastNodeId >= 0) {
                    links.push({
                        source: lastNodeId,
                        target: nodeId
                    });
                }
                lastNodeId = nodeId;
                nodeId++;
            }
        }
    });
}

function visualizeReActAgent(messages, nodes, links, nodeId, lastNodeId) {
    // Process ReAct agent messages
    // In ReAct, tool results appear in messages with role='user' starting with 'Observation:'
    messages.forEach((msg, idx) => {
        const role = msg.role;
        const content = msg.content || '';

        // Check if this is a tool observation message
        const isToolObservation = role === 'user' && content.startsWith('Observation:');

        if (currentFilter === 'all') {
            // ALL MODE: Show all messages
            if (isToolObservation) {
                // Extract tool content from observation
                const toolContent = extractToolContent(content);

                nodes.push({
                    id: nodeId,
                    type: 'tool',
                    content: toolContent.result,
                    timestamp: msg.timestamp || '',
                    index: idx,
                    toolName: toolContent.tool_name || msg.name || 'tool',
                    arguments: toolContent.arguments,
                    status: toolContent.status,
                    duration: toolContent.duration,
                    error_message: toolContent.error_message
                });
            } else {
                nodes.push({
                    id: nodeId,
                    type: role,
                    content: content,
                    timestamp: msg.timestamp || '',
                    index: idx
                });
            }

            if (nodeId > 0) {
                links.push({
                    source: nodeId - 1,
                    target: nodeId
                });
            }
            nodeId++;
        } else if (currentFilter === 'tools') {
            // TOOLS ONLY MODE: Show only tool observations
            if (isToolObservation) {
                const toolContent = extractToolContent(content);

                nodes.push({
                    id: nodeId,
                    type: 'tool',
                    content: toolContent.result,
                    timestamp: msg.timestamp || '',
                    index: idx,
                    toolName: toolContent.tool_name || msg.name || 'tool',
                    arguments: toolContent.arguments,
                    status: toolContent.status,
                    duration: toolContent.duration,
                    error_message: toolContent.error_message
                });

                // Connect to previous tool node
                if (lastNodeId >= 0) {
                    links.push({
                        source: lastNodeId,
                        target: nodeId
                    });
                }
                lastNodeId = nodeId;
                nodeId++;
            }
        } else if (currentFilter === 'no-tools') {
            // NO TOOLS MODE: Show all messages except tool observations
            if (!isToolObservation) {
                nodes.push({
                    id: nodeId,
                    type: role,
                    content: content,
                    timestamp: msg.timestamp || '',
                    index: idx
                });

                // Connect to previous non-tool node
                if (lastNodeId >= 0) {
                    links.push({
                        source: lastNodeId,
                        target: nodeId
                    });
                }
                lastNodeId = nodeId;
                nodeId++;
            }
        }
    });
}

function extractToolContent(observationText) {
    // Extract tool content from "Observation: {tool_content}" format
    try {
        // Remove "Observation: " prefix
        let contentStr = observationText.replace(/^Observation:\s*/, '');

        // Try to parse as JSON
        let toolData = null;
        try {
            // Try direct JSON parse first
            toolData = JSON.parse(contentStr);
        } catch (e1) {
            // If that fails, it might be a Python dict string representation
            // Convert Python dict syntax to JSON:
            // 1. Replace Python None with null
            // 2. Replace Python True/False with true/false
            // 3. Replace single quotes with double quotes
            let jsonStr = contentStr
                .replace(/None/g, 'null')
                .replace(/True/g, 'true')
                .replace(/False/g, 'false')
                // Replace single quotes with double quotes
                // This is a simple approach that works for most cases
                .replace(/'/g, '"');

            toolData = JSON.parse(jsonStr);
        }

        return {
            tool_name: toolData.tool_name,
            arguments: toolData.arguments,
            result: toolData.result,
            status: toolData.status,
            error_message: toolData.error_message,
            duration: toolData.duration
        };
    } catch (e) {
        // If parsing fails, return the raw content
        console.log('Failed to parse observation:', e);
        return {
            result: observationText
        };
    }
}

function drawGraph(nodes, links) {
    // Clear existing graph
    d3.select('#graph-container').selectAll('*').remove();

    if (nodes.length === 0) {
        d3.select('#graph-container').append('div')
            .attr('class', 'no-data')
            .html('<h3>No nodes to display</h3><p>Try changing the filter mode</p>');
        return;
    }

    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg = d3.select('#graph-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Define arrow marker
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 25)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('fill', '#999');

    g = svg.append('g');

    // Add zoom behavior
    zoomBehavior = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    svg.call(zoomBehavior);

    // Group nodes by tool name for parallel positioning
    const toolGroups = new Map();
    nodes.forEach(node => {
        if (node.toolName) {
            if (!toolGroups.has(node.toolName)) {
                toolGroups.set(node.toolName, []);
            }
            toolGroups.get(node.toolName).push(node);
        }
    });

    // Set initial positions based on execution order (left to right)
    const horizontalSpacing = Math.max(100, (width - 100) / Math.max(1, nodes.length - 1));
    nodes.forEach((node, i) => {
        node.x = 50 + i * horizontalSpacing;
        node.y = height / 2;
    });

    // Create force simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('x', d3.forceX(d => 50 + d.id * horizontalSpacing).strength(0.3)) // Strong x force to maintain left-to-right order
        .force('y', d3.forceY(height / 2).strength(0.05)) // Weak y force to center vertically
        .force('collision', d3.forceCollide().radius(50))

    // Draw links
    const link = g.append('g')
        .selectAll('path')
        .data(links)
        .join('path')
        .attr('class', 'link')
        .attr('stroke', '#999');

    // Draw nodes
    const node = g.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .on('click', showDetails);

    node.append('circle')
        .attr('r', 20)
        .attr('fill', d => getNodeColor(d.type))
        .attr('stroke', '#333')
        .attr('opacity', d => {
            // Dim nodes that cannot be annotated yet
            if (d.type === 'system' || d.type === 'tool') {
                return 1.0; // Keep system and tool nodes at full opacity
            }
            return isNodeAnnotatable(d) ? 1.0 : 0.4;
        })
        .attr('stroke-dasharray', d => {
            // Add dashed border for nodes that need previous nodes to be labeled
            if (d.type === 'system' || d.type === 'tool') {
                return null;
            }
            // Check if this would be annotatable if all previous nodes were labeled
            const wouldBeAnnotatable = d.type !== 'system' &&
                                      d.type !== 'tool' &&
                                      !(d.type === 'user' && currentNodes.find(n => n.type === 'user')?.id === d.id);
            return (!isNodeAnnotatable(d) && wouldBeAnnotatable) ? '5,5' : null;
        });

    node.append('text')
        .attr('dy', 35)
        .attr('text-anchor', 'middle')
        .text(d => getNodeLabel(d))
        .style('fill', '#333');

    // Update positions on simulation tick
    simulation.on('tick', () => {
        link.attr('d', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy);
            return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
        });

        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function getNodeColor(type) {
    const typeMap = {
        'system': colors.system,
        'user': colors.user,
        'assistant': colors.assistant,
        'tool': colors.result,
        'tool_call': colors.tool
    };
    return typeMap[type] || '#6c757d';
}

function getNodeLabel(node) {
    if (node.type === 'tool' || node.type === 'tool_call') {
        return node.toolName ? node.toolName.substring(0, 15) : 'tool';
    }
    return node.type;
}

function updateNodeAnnotatability() {
    // Update the visual appearance of all nodes based on their annotatable state
    if (!svg) return;

    d3.selectAll('.node circle')
        .attr('opacity', d => {
            // Dim nodes that cannot be annotated yet
            if (d.type === 'system' || d.type === 'tool') {
                return 1.0; // Keep system and tool nodes at full opacity
            }
            return isNodeAnnotatable(d) ? 1.0 : 0.4;
        })
        .attr('stroke-dasharray', d => {
            // Add dashed border for nodes that need previous nodes to be labeled
            if (d.type === 'system' || d.type === 'tool') {
                return null;
            }
            // Check if this would be annotatable if all previous nodes were labeled
            const wouldBeAnnotatable = d.type !== 'system' &&
                                      d.type !== 'tool' &&
                                      !(d.type === 'user' && currentNodes.find(n => n.type === 'user')?.id === d.id);
            return (!isNodeAnnotatable(d) && wouldBeAnnotatable) ? '5,5' : null;
        });
}

function showDetails(event, d) {
    const panel = document.getElementById('detailsPanel');
    const content = document.getElementById('detailsContent');

    // Update selected node index
    selectedNodeIndex = currentNodes.findIndex(node => node.id === d.id);

    // Update visual selection
    d3.selectAll('.node circle')
        .attr('stroke-width', node => node.id === d.id ? 4 : 1)
        .attr('stroke', node => node.id === d.id ? '#ff6b6b' : '#333');

    // Update button states
    updateNodeNavButtons();

    // Check if this node is annotatable
    const annotatable = isNodeAnnotatable(d);

    // Enable/disable marker dropdowns based on whether the node is annotatable
    document.getElementById('neutralMarkerSelect').disabled = !annotatable;
    document.getElementById('positiveMarkerSelect').disabled = !annotatable;
    document.getElementById('negativeMarkerSelect').disabled = !annotatable;

    // Enable/disable marker buttons (for backwards compatibility) and notes textarea based on whether the node is annotatable
    document.querySelectorAll('.marker-btn').forEach(btn => {
        btn.disabled = !annotatable;
    });
    document.getElementById('notesTextarea').disabled = !annotatable;

    // Show/hide disabled message
    const disabledMsg = document.getElementById('annotationDisabledMsg');
    if (disabledMsg) {
        if (annotatable) {
            disabledMsg.style.display = 'none';
        } else {
            disabledMsg.style.display = 'block';
            disabledMsg.textContent = getAnnotationDisabledReason(d);
        }
    }

    // Load annotations for this node (or clear if not annotatable)
    if (annotatable) {
        loadNodeAnnotations(d.id);
    } else {
        // Clear annotations display for non-annotatable nodes
        document.getElementById('selectedMarkers').innerHTML = '';
        document.getElementById('notesTextarea').value = '';
        // Clear button selections (for backwards compatibility)
        document.querySelectorAll('.marker-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
    }

    let html = '';

    html += `<div class="detail-item">
        <div class="detail-label">Node Type</div>
        <div class="detail-value">${d.type}</div>
    </div>`;

    html += `<div class="detail-item">
        <div class="detail-label">Message Index</div>
        <div class="detail-value">${d.index}</div>
    </div>`;

    if (d.toolName) {
        html += `<div class="detail-item">
            <div class="detail-label">Tool Name</div>
            <div class="detail-value">${d.toolName}</div>
        </div>`;
    }

    if (d.status) {
        html += `<div class="detail-item">
            <div class="detail-label">Status</div>
            <div class="detail-value">${d.status}</div>
        </div>`;
    }

    if (d.duration !== undefined) {
        html += `<div class="detail-item">
            <div class="detail-label">Duration</div>
            <div class="detail-value">${d.duration.toFixed(4)} seconds</div>
        </div>`;
    }

    if (d.error_message) {
        html += `<div class="detail-item">
            <div class="detail-label">Error Message</div>
            <div class="detail-value"><pre>${d.error_message}</pre></div>
        </div>`;
    }

    if (d.arguments) {
        // Try to parse if it's a string, otherwise use as-is
        let argsToDisplay = d.arguments;
        if (typeof d.arguments === 'string') {
            try {
                argsToDisplay = JSON.parse(d.arguments);
            } catch (e) {
                argsToDisplay = d.arguments;
            }
        }
        html += `<div class="detail-item">
            <div class="detail-label">Arguments</div>
            <div class="detail-value"><pre>${typeof argsToDisplay === 'object' ? JSON.stringify(argsToDisplay, null, 2) : argsToDisplay}</pre></div>
        </div>`;
    }

    if (d.content) {
        // Try to parse content as JSON/dict and display each key separately
        let contentObj = null;

        // First, try to parse the string content as JSON
        try {
            if (typeof d.content === 'string') {
                // Try direct JSON parse first
                try {
                    contentObj = JSON.parse(d.content);
                } catch (e1) {
                    // If that fails, it might be a Python dict string representation
                    // Use a more robust approach to convert Python dict to JSON
                    try {
                        // Replace Python literals with JSON equivalents
                        let jsonStr = d.content
                            .replace(/\bNone\b/g, 'null')
                            .replace(/\bTrue\b/g, 'true')
                            .replace(/\bFalse\b/g, 'false');

                        // Replace single quotes with double quotes, being careful about escaped quotes
                        // This regex handles most cases including strings with escaped single quotes
                        jsonStr = jsonStr.replace(/(\w+):\s*'([^']*)'/g, '"$1": "$2"')  // key: 'value'
                                       .replace(/{\s*'([^']+)':/g, '{"$1":')           // {'key':
                                       .replace(/,\s*'([^']+)':/g, ', "$1":')          // , 'key':
                                       .replace(/:\s*'([^']*)'/g, ': "$1"');           // : 'value'

                        contentObj = JSON.parse(jsonStr);
                    } catch (e2) {
                        // Try using eval as last resort (with Function constructor for safety)
                        try {
                            // Create a safe evaluation context
                            const evalFunc = new Function('return ' + d.content.replace(/\bNone\b/g, 'null')
                                                                               .replace(/\bTrue\b/g, 'true')
                                                                               .replace(/\bFalse\b/g, 'false'));
                            contentObj = evalFunc();
                        } catch (e3) {
                            console.log('All parsing attempts failed:', e3);
                            contentObj = null;
                        }
                    }
                }
            } else {
                contentObj = d.content;
            }
        } catch (e) {
            // If all parsing fails, content is plain text
            console.log('Failed to parse content:', e);
            contentObj = null;
        }

        if (contentObj && typeof contentObj === 'object') {
            // For tool role nodes, use the same boxed formatting as tool_calls
            if (d.type === 'tool') {
                html += `<div class="detail-item">
                    <div class="detail-label">Tool Response</div>
                    <div class="detail-value">
                        <div class="tool-call-box" style="background-color: #f8f9fa; border-left: 4px solid #17a2b8; padding: 12px; border-radius: 4px;">`;

                // Display tool_name if present
                if (contentObj.tool_name !== undefined) {
                    html += `<div style="margin-bottom: 6px;"><strong>Tool Name:</strong> <span style="color: #fd7e14; font-weight: 500;">${contentObj.tool_name}</span></div>`;
                }

                // Display status if present
                if (contentObj.status !== undefined) {
                    const statusColor = contentObj.status === 'success' ? '#28a745' : '#dc3545';
                    html += `<div style="margin-bottom: 6px;"><strong>Status:</strong> <span style="color: ${statusColor}; font-weight: 500;">${contentObj.status}</span></div>`;
                }

                // Display duration if present
                if (contentObj.duration !== undefined) {
                    const durationValue = typeof contentObj.duration === 'number'
                        ? contentObj.duration.toFixed(4)
                        : contentObj.duration;
                    html += `<div style="margin-bottom: 6px;"><strong>Duration:</strong> <span style="color: #495057;">${durationValue} seconds</span></div>`;
                }

                // Display timestamp if present
                if (contentObj.timestamp !== undefined) {
                    html += `<div style="margin-bottom: 6px;"><strong>Timestamp:</strong> <span style="color: #6c757d; font-size: 0.9em;">${contentObj.timestamp}</span></div>`;
                }

                // Display arguments if present
                if (contentObj.arguments !== undefined) {
                    const argsValue = typeof contentObj.arguments === 'object'
                        ? JSON.stringify(contentObj.arguments, null, 2)
                        : contentObj.arguments;
                    html += `<div style="margin-bottom: 6px;"><strong>Arguments:</strong><pre style="background-color: #ffffff; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto;">${argsValue}</pre></div>`;
                }

                // Display result if present
                if (contentObj.result !== undefined) {
                    const resultValue = typeof contentObj.result === 'object'
                        ? JSON.stringify(contentObj.result, null, 2)
                        : contentObj.result;
                    html += `<div style="margin-bottom: 6px;"><strong>Result:</strong><pre style="background-color: #ffffff; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto;">${resultValue}</pre></div>`;
                }

                // Display error_message if present
                if (contentObj.error_message !== undefined && contentObj.error_message !== null) {
                    html += `<div style="margin-bottom: 6px;"><strong>Error Message:</strong><pre style="background-color: #fff3cd; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto; color: #856404;">${contentObj.error_message}</pre></div>`;
                }

                // Display any other keys that might exist (excluding the ones we've already shown)
                const displayedKeys = ['tool_name', 'status', 'duration', 'timestamp', 'arguments', 'result', 'error_message'];
                Object.keys(contentObj).forEach(key => {
                    if (!displayedKeys.includes(key)) {
                        const value = contentObj[key];
                        const displayValue = typeof value === 'object'
                            ? `<pre style="background-color: #ffffff; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto;">${JSON.stringify(value, null, 2)}</pre>`
                            : `<span style="color: #495057;">${value}</span>`;
                        html += `<div style="margin-bottom: 6px;"><strong>${key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}:</strong> ${displayValue}</div>`;
                    }
                });

                html += `</div></div></div>`;
            } else {
                // For non-tool roles, use the original formatting
                if (contentObj.arguments !== undefined) {
                    html += `<div class="detail-item">
                        <div class="detail-label">Arguments</div>
                        <div class="detail-value"><pre>${JSON.stringify(contentObj.arguments, null, 2)}</pre></div>
                    </div>`;
                }

                if (contentObj.result !== undefined) {
                    html += `<div class="detail-item">
                        <div class="detail-label">Result</div>
                        <div class="detail-value"><pre>${JSON.stringify(contentObj.result, null, 2)}</pre></div>
                    </div>`;
                }

                if (contentObj.status !== undefined) {
                    html += `<div class="detail-item">
                        <div class="detail-label">Status</div>
                        <div class="detail-value">${contentObj.status}</div>
                    </div>`;
                }

                // Display any other keys that might exist
                Object.keys(contentObj).forEach(key => {
                    if (key !== 'arguments' && key !== 'result' && key !== 'status') {
                        const value = contentObj[key];
                        html += `<div class="detail-item">
                            <div class="detail-label">${key.charAt(0).toUpperCase() + key.slice(1)}</div>
                            <div class="detail-value">${typeof value === 'object' ? `<pre>${JSON.stringify(value, null, 2)}</pre>` : value}</div>
                        </div>`;
                    }
                });
            }
        } else {
            // If content is not JSON or parsing failed, display as plain text in a scrollable container
            html += `<div class="detail-item">
                <div class="detail-label">Content</div>
                <div class="detail-value"><pre>${d.content}</pre></div>
            </div>`;
        }
    }

    // Display tool_calls if present (after content)
    if (d.toolCalls && Array.isArray(d.toolCalls) && d.toolCalls.length > 0) {
        html += `<div class="detail-item">
            <div class="detail-label">Tool Calls</div>
            <div class="detail-value">`;

        d.toolCalls.forEach((toolCall, idx) => {
            html += `<div class="tool-call-box" style="background-color: #f8f9fa; border-left: 4px solid #667eea; padding: 12px; margin-bottom: 8px; border-radius: 4px;">`;
            html += `<div style="font-weight: 600; color: #667eea; margin-bottom: 8px;">Tool Call ${idx + 1}</div>`;

            if (toolCall.id) {
                html += `<div style="margin-bottom: 6px;"><strong>ID:</strong> <code style="background-color: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">${toolCall.id}</code></div>`;
            }

            if (toolCall.function && toolCall.function.name) {
                html += `<div style="margin-bottom: 6px;"><strong>Function:</strong> <span style="color: #fd7e14; font-weight: 500;">${toolCall.function.name}</span></div>`;
            }

            if (toolCall.function && toolCall.function.arguments) {
                let argsToDisplay = toolCall.function.arguments;
                try {
                    if (typeof argsToDisplay === 'string') {
                        argsToDisplay = JSON.parse(argsToDisplay);
                    }
                    html += `<div style="margin-bottom: 6px;"><strong>Arguments:</strong><pre style="background-color: #ffffff; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto;">${JSON.stringify(argsToDisplay, null, 2)}</pre></div>`;
                } catch (e) {
                    html += `<div style="margin-bottom: 6px;"><strong>Arguments:</strong><pre style="background-color: #ffffff; padding: 8px; border-radius: 3px; margin-top: 4px; font-size: 0.85em; max-height: 300px; overflow-y: auto;">${argsToDisplay}</pre></div>`;
                }
            }

            html += `</div>`;
        });

        html += `</div></div>`;
    }

    if (d.toolCallId) {
        html += `<div class="detail-item">
            <div class="detail-label">Tool Call ID</div>
            <div class="detail-value">${d.toolCallId}</div>
        </div>`;
    }

    content.innerHTML = html;
    // Panel is always visible now, so no need to change display
}

function addLegend() {
    const legend = d3.select('#graph-container')
        .append('div')
        .attr('class', 'legend');

    legend.append('div')
        .attr('class', 'legend-title')
        .text('Node Types');

    const items = [
        { type: 'system', label: 'System' },
        { type: 'user', label: 'User' },
        { type: 'assistant', label: 'Assistant' },
        { type: 'tool_call', label: 'Tool Call' },
        { type: 'tool', label: 'Tool Result' }
    ];

    items.forEach(item => {
        const legendItem = legend.append('div')
            .attr('class', 'legend-item');

        legendItem.append('div')
            .attr('class', 'legend-color')
            .style('background-color', getNodeColor(item.type));

        legendItem.append('span')
            .text(item.label);
    });
}

function dragstarted(event, d, toolGroups) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d, toolGroups) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d, toolGroups) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Submit button handler
document.getElementById('submitBtn').addEventListener('click', async function() {
    // Save current file's annotations and trace comments
    if (allFiles.length > 0 && allFiles[currentFileIndex]) {
        const currentFileName = allFiles[currentFileIndex].name;
        allFileAnnotations[currentFileName] = JSON.parse(JSON.stringify(nodeAnnotations));
        const traceCommentsTextarea = document.getElementById('traceCommentsTextarea');
        allFileTraceComments[currentFileName] = traceCommentsTextarea.value;
    }

    // Validate annotator name
    const annotatorName = document.getElementById('annotatorName').value.trim();
    if (!annotatorName) {
        alert('❌ Validation Error: Annotator Name is required.');
        document.getElementById('annotatorName').focus();
        return;
    }

    // Validate Personal Identifier
    const mongodbKey = document.getElementById('mongodbKey').value.trim();
    if (!mongodbKey) {
        alert('❌ Validation Error: Personal Identifier is required.');
        document.getElementById('mongodbKey').focus();
        return;
    }

    // Check all files for missing annotations
    const missingAnnotations = [];

    allFiles.forEach((file, fileIndex) => {
        const fileName = file.name;
        const fileAnnotations = allFileAnnotations[fileName] || {};
        const fileNodes = allFileNodes[fileName] || [];

        // Check if file has been loaded (has node data)
        if (fileNodes.length === 0) {
            missingAnnotations.push({
                fileName: fileName,
                fileIndex: fileIndex + 1,
                reason: 'File has not been loaded yet'
            });
            return;
        }

        // Check all annotatable nodes in this file
        const unannotatedInFile = [];
        fileNodes.forEach((nodeInfo, nodeIndex) => {
            if (nodeInfo.annotatable) {
                const annotation = fileAnnotations[nodeInfo.id];
                if (!annotation || annotation.markers.length === 0) {
                    unannotatedInFile.push({
                        nodeId: nodeInfo.id,
                        nodeIndex: nodeIndex,
                        nodeType: nodeInfo.type
                    });
                }
            }
        });

        if (unannotatedInFile.length > 0) {
            missingAnnotations.push({
                fileName: fileName,
                fileIndex: fileIndex + 1,
                unannotatedNodes: unannotatedInFile
            });
        }
    });

    // If there are missing annotations, show detailed error
    if (missingAnnotations.length > 0) {
        let message = '❌ Validation Error: Not all annotatable nodes have been marked.\n\n';

        missingAnnotations.forEach(item => {
            message += `📄 File ${item.fileIndex}: ${item.fileName}\n`;

            if (item.reason) {
                message += `   ${item.reason}\n`;
            } else if (item.unannotatedNodes) {
                message += `   Missing markers on ${item.unannotatedNodes.length} node(s):\n`;
                item.unannotatedNodes.slice(0, 5).forEach(node => {
                    message += `   • Node ${node.nodeIndex} (${node.nodeType})\n`;
                });
                if (item.unannotatedNodes.length > 5) {
                    message += `   ... and ${item.unannotatedNodes.length - 5} more\n`;
                }
            }
            message += '\n';
        });

        message += 'Please review all files and ensure all annotatable nodes have markers assigned.';
        alert(message);
        return;
    }

    // Calculate total annotations
    let totalMarkers = 0;
    let totalNotes = 0;
    Object.values(allFileAnnotations).forEach(fileAnnotations => {
        Object.values(fileAnnotations).forEach(annotation => {
            totalMarkers += annotation.markers.length;
            if (annotation.notes && annotation.notes.trim()) {
                totalNotes++;
            }
        });
    });

    // Prepare the payload for the API
    const payload = {
        annotator: annotatorName,
        mongodbKey: mongodbKey,
        annotations: allFileAnnotations,
        traceComments: allFileTraceComments,
        fileNodes: allFileNodes
    };

    // Show submitting message
    const submitBtn = document.getElementById('submitBtn');
    const originalBtnText = submitBtn.textContent;
    submitBtn.textContent = '⏳ Submitting...';
    submitBtn.disabled = true;

    try {
        // Send data to the API endpoint
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            // Success
            const totalFiles = allFiles.length;
            let successMessage = '✅ Submission Successful!\n\n';
            successMessage += `Annotator: ${annotatorName}\n`;
            successMessage += `Personal Identifier: ${mongodbKey}\n`;
            successMessage += `Files Processed: ${totalFiles}/${totalFiles}\n`;
            successMessage += `Total Markers: ${totalMarkers}\n`;
            successMessage += `Nodes with Notes: ${totalNotes}\n\n`;
            successMessage += `New Entries Inserted: ${result.inserted}`;

            alert(successMessage);

            // Log successful submission
            console.log('Submission successful:', result);

            // Clear cache after successful submission
            clearCachedData();
        } else {
            // API returned an error
            let errorMessage = '❌ Submission Failed\n\n';
            errorMessage += `Status: ${response.status}\n`;
            errorMessage += `Error: ${result.detail || JSON.stringify(result)}`;
            alert(errorMessage);
            console.error('Submission error:', result);
        }
    } catch (error) {
        // Network or other error
        let errorMessage = '❌ Submission Failed\n\n';
        errorMessage += `Error: ${error.message}\n\n`;
        errorMessage += 'Please check:\n';
        errorMessage += '• Network connection\n';
        errorMessage += '• API endpoint availability';
        alert(errorMessage);
        console.error('Submission exception:', error);
    } finally {
        // Restore button state
        submitBtn.textContent = originalBtnText;
        submitBtn.disabled = false;
    }
});

// ============================================================================
// Cache Management Functions
// ============================================================================

/**
 * Generate cache key - now simplified to a single key for all users
 */
function getCacheKey() {
    return CACHE_KEY;
}

/**
 * Save current annotations to localStorage
 * Automatically called when switching files or making annotations
 */
function saveCachedData() {
    const cacheKey = getCacheKey();

    try {
        const cacheData = {
            version: CACHE_VERSION,
            timestamp: Date.now(),
            annotations: allFileAnnotations,
            nodes: allFileNodes,
            traceComments: allFileTraceComments,
            currentFileIndex: currentFileIndex,
            fileNames: allFiles.map(f => f.name)
        };

        localStorage.setItem(cacheKey, JSON.stringify(cacheData));
        console.log(`✓ Annotations cached successfully`);
    } catch (error) {
        console.warn('Failed to cache annotations:', error);
        // localStorage quota exceeded or unavailable - fail silently
    }
}

/**
 * Load cached annotations from localStorage
 * Called on page load and when files are loaded
 */
function loadCachedData() {
    const cacheKey = getCacheKey();

    try {
        const cached = localStorage.getItem(cacheKey);
        if (!cached) {
            console.log('ℹ No cached data found');
            return false;
        }

        const cacheData = JSON.parse(cached);

        // Validate cache version
        if (cacheData.version !== CACHE_VERSION) {
            console.log('Cache version mismatch, ignoring cached data');
            localStorage.removeItem(cacheKey);
            return false;
        }

        // Only load if we have files loaded and they match the cached file names
        if (allFiles.length > 0) {
            const currentFileNames = allFiles.map(f => f.name);
            const cachedFileNames = cacheData.fileNames || [];

            console.log(`Checking cache: current=${currentFileNames.length} files, cached=${cachedFileNames.length} files`);

            // Check if file sets match (same files loaded)
            if (JSON.stringify(currentFileNames.sort()) === JSON.stringify(cachedFileNames.sort())) {
                allFileAnnotations = cacheData.annotations || {};
                allFileNodes = cacheData.nodes || {};
                allFileTraceComments = cacheData.traceComments || {};

                // Restore current file's annotations
                const currentFileName = allFiles[currentFileIndex].name;
                nodeAnnotations = allFileAnnotations[currentFileName] || {};

                // Restore trace comments
                const traceCommentsTextarea = document.getElementById('traceCommentsTextarea');
                if (traceCommentsTextarea) {
                    traceCommentsTextarea.value = allFileTraceComments[currentFileName] || '';
                }

                // Refresh the visualization to show annotated nodes
                updateNodeAnnotatability();

                // If a node is selected, update its details
                if (selectedNodeIndex >= 0 && currentNodes[selectedNodeIndex]) {
                    loadNodeAnnotations(currentNodes[selectedNodeIndex].id);
                }

                console.log(`✓ Loaded cached annotations (${Object.keys(allFileAnnotations).length} files)`);
                showCacheNotification('Restored previous annotations from cache');
                return true;
            } else {
                console.log('⚠ File sets do not match - cache not loaded');
            }
        }

        return false;
    } catch (error) {
        console.warn('Failed to load cached annotations:', error);
        return false;
    }
}

/**
 * Clear cached data for current session
 */
function clearCachedData() {
    const cacheKey = getCacheKey();
    if (cacheKey) {
        try {
            localStorage.removeItem(cacheKey);
            console.log('✓ Cache cleared');
            showCacheNotification('Cache cleared successfully');
        } catch (error) {
            console.warn('Failed to clear cache:', error);
        }
    }
}

/**
 * Show a temporary notification about cache operations
 */
function showCacheNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'cache-notification';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 12px 24px;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        font-size: 14px;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Auto-save wrapper for annotation changes
 */
function autoSaveAnnotations() {
    // Save current file's annotations
    if (allFiles.length > 0 && allFiles[currentFileIndex]) {
        const currentFileName = allFiles[currentFileIndex].name;
        allFileAnnotations[currentFileName] = JSON.parse(JSON.stringify(nodeAnnotations));
        const traceCommentsTextarea = document.getElementById('traceCommentsTextarea');
        allFileTraceComments[currentFileName] = traceCommentsTextarea.value;
    }

    // Save to cache
    saveCachedData();
}

/**
 * Check if there's a cached session available on page load
 */
function checkForCachedSession() {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) {
            return;
        }

        const cachedData = JSON.parse(cached);
        if (cachedData && cachedData.fileNames && cachedData.fileNames.length > 0) {
            const fileCount = cachedData.fileNames.length;
            const timestamp = new Date(cachedData.timestamp).toLocaleString();

            // Show persistent notification about cached session
            showPersistentCacheNotification(
                `💾 Cached session found!\n` +
                `${fileCount} files, last saved: ${timestamp}\n` +
                `Load the same files to restore your annotations.`,
                cachedData.fileNames
            );
        }
    } catch (error) {
        console.warn('Error checking for cached session:', error);
    }
}

/**
 * Show a persistent notification with file list
 */
function showPersistentCacheNotification(message, fileNames) {
    const notification = document.createElement('div');
    notification.className = 'cache-notification persistent';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.4s ease-out;
        font-size: 14px;
        max-width: 450px;
        line-height: 1.6;
        border: 2px solid rgba(255,255,255,0.3);
    `;

    const lines = message.split('\n');
    let html = `<div style="margin-bottom: 10px; font-size: 16px;"><strong>✨ ${lines[0]}</strong></div>`;
    for (let i = 1; i < lines.length; i++) {
        if (lines[i]) {
            html += `<div style="font-size: 13px; opacity: 0.95; margin-top: 4px;">${lines[i]}</div>`;
        }
    }

    if (fileNames && fileNames.length > 0) {
        html += `<details style="margin-top: 12px; font-size: 12px; cursor: pointer;">
            <summary style="cursor: pointer; opacity: 0.9; user-select: none; padding: 4px 0;">
                📁 Show cached files (${fileNames.length})
            </summary>
            <ul style="margin: 8px 0 0 0; padding-left: 20px; max-height: 200px; overflow-y: auto; background: rgba(0,0,0,0.1); border-radius: 4px; padding: 8px 8px 8px 24px;">
                ${fileNames.slice(0, 20).map(f => `<li style="margin: 4px 0;">${f}</li>`).join('')}
                ${fileNames.length > 20 ? `<li style="margin: 4px 0; opacity: 0.8;"><em>... and ${fileNames.length - 20} more</em></li>` : ''}
            </ul>
        </details>`;
    }

    html += `<button style="
        margin-top: 16px;
        padding: 8px 16px;
        background: rgba(255,255,255,0.25);
        border: 1px solid rgba(255,255,255,0.5);
        color: white;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        transition: all 0.2s;
    " onmouseover="this.style.background='rgba(255,255,255,0.35)'"
       onmouseout="this.style.background='rgba(255,255,255,0.25)'"
       onclick="this.parentElement.remove()">Got it, dismiss</button>`;

    notification.innerHTML = html;
    document.body.appendChild(notification);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
