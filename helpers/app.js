let allFiles = [];
let currentFileIndex = 0;
let currentFilter = 'all';
let svg, g, simulation, zoomBehavior;
let currentNodes = []; // Store current graph nodes
let selectedNodeIndex = -1; // Track selected node index (-1 means none selected)

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

function visualizeTrace(data) {
    const agentType = data.agent || 'ToolCallingAgent'; // Default to ToolCallingAgent
    const messages = data.messages;
    const nodes = [];
    const links = [];
    let nodeId = 0;
    let lastNodeId = -1;

    // Reset node selection when loading new trace
    selectedNodeIndex = -1;

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
                toolName: msg.name
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
                    toolName: msg.name
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
                    toolName: msg.name
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
        .attr('stroke', '#333');

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
                    // Convert Python dict syntax to JSON:
                    // 1. Replace single quotes with double quotes (but not inside strings)
                    // 2. Replace Python None with null
                    // 3. Replace Python True/False with true/false
                    let jsonStr = d.content
                        .replace(/None/g, 'null')
                        .replace(/True/g, 'true')
                        .replace(/False/g, 'false')
                        // Replace single quotes with double quotes
                        // This is a simple approach that works for most cases
                        .replace(/'/g, '"');

                    contentObj = JSON.parse(jsonStr);
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
            // Display each key from the content dict
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
        } else {
            // If content is not JSON or parsing failed, display as plain text in a scrollable container
            html += `<div class="detail-item">
                <div class="detail-label">Content</div>
                <div class="detail-value"><pre>${d.content}</pre></div>
            </div>`;
        }
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
