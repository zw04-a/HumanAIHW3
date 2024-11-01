const sendBtn = document.querySelector('#send-btn');
const promptInput = document.querySelector('#prompt-input');
const responseText = document.querySelector('#response-text');
const uploadArea = document.querySelector('.upload_area');
const previewBtn = document.querySelector('#preview-btn');
const clearMessagesBtn = document.querySelector('#clear-messages-btn');



let datasetUploaded = false;  // Flag to check if a dataset is uploaded
let previewVisible = false;   // Flag to track if the preview window is visible
let previewData = [];         // Variable to hold the preview data
let parsedData = [];          // Global variable to store the complete dataset

// Add event listener
clearMessagesBtn.addEventListener('click', function() {
    const messagesContainer = document.querySelector('#response-text');
    while (messagesContainer.firstChild) {
        messagesContainer.removeChild(messagesContainer.firstChild);
    }
});

// Enable or disable the send button based on input value
promptInput.addEventListener('input', function(event) {
    sendBtn.disabled = event.target.value ? false : true;
});

// Handle file upload
uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.style.borderColor = '#337ab7'; // Highlight area on drag over
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#ccc'; // Revert border color on drag leave
});

uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.style.borderColor = '#ccc'; // Revert border color
    const file = event.dataTransfer.files[0];
    handleFileUpload(file);
});

uploadArea.addEventListener('click', () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.csv';
    fileInput.onchange = (e) => {
        const file = e.target.files[0];
        handleFileUpload(file);
    };
    fileInput.click();
});

// Helper function to get column names from the preview data
function getColumnNames(previewData) {
    if (previewData.length > 0) {
        return Object.keys(previewData[0]);
    }
    return [];
}

// Helper function to determine data types (including all four types)
function getDataTypes(previewData) {
    if (previewData.length > 0) {
        const types = {};
        const firstRow = previewData[0];
        const columns = Object.keys(firstRow);

        columns.forEach(column => {
            const sampleValues = previewData.map(row => row[column]);
            let type = 'nominal'; // Default to nominal

            // Determine if the column is temporal (date or time)
            if (sampleValues.every(value => !isNaN(Date.parse(value)))) {
                type = 'temporal';
            }
            // Determine if the column is quantitative (all values are numbers)
            else if (sampleValues.every(value => !isNaN(value) && typeof value === 'number')) {
                type = 'quantitative';
            }
            // Determine if the column is ordinal (discrete ordered categories)
            else if (sampleValues.every(value => typeof value === 'string' && !isNaN(parseFloat(value)))) {
                type = 'ordinal';
            }

            types[column] = type;
        });

        return types;
    }
    return {};
}

// Helper function to get sample values (first few rows)
function getSampleValues(previewData, numberOfSamples = 3) {
    return previewData.slice(0, numberOfSamples);
}

// Function to generate the Vega-Lite prompt
function generateVegaLitePrompt(columns, types, sampleValues, userQuery) {
    let prompt = `You are to generate a JSON object that adheres to the Vega-Lite specification to create a data visualization.\n`;
    prompt += `The dataset has the following columns: ${columns.join(", ")}.\n`;
    prompt += `The data types for each column are: ${JSON.stringify(types)}.\n`;
    prompt += `Here are some sample values from the dataset: ${JSON.stringify(sampleValues)}.\n`;
    prompt += `Please generate a Vega-Lite JSON specification to visualize this data based on the following user query: ${userQuery}.\n`;
    prompt += `Ensure the JSON output is structured with proper "data", "mark", "encoding" keys as per Vega-Lite standard.\n`;
    prompt += `Use "https://vega.github.io/schema/vega-lite/v5.json" as the schema URL.`;

    return prompt;
}


function cleanupExistingPreview() {
    const existingPreviews = document.querySelectorAll('.preview-window');
    existingPreviews.forEach(preview => {
        preview.remove();
    });
    previewVisible = false;
    previewBtn.textContent = "Show Table Preview";
}

function handleFileUpload(file) {
    if (file && file.type === 'text/csv') {
        // Clean up any existing preview first
        cleanupExistingPreview();
        
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                datasetUploaded = true;
                fetchPreviewData().then(() => {
                    if (previewData.length > 0) {
                        showPreviewWindow();
                    } else {
                        addMessageToChat('left', 'No preview available.');
                    }
                });

                const reader = new FileReader();
                reader.onload = function(event) {
                    const csvData = event.target.result;
                    parsedData = d3.csvParse(csvData, d3.autoType);
                };
                reader.readAsText(file);
            } else {
                addMessageToChat('left', 'Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChat('left', 'Error: Failed to upload the dataset.');
        });
    } else {
        addMessageToChat('left', 'Please upload a valid CSV file.');
    }
}

function fetchPreviewData() {
    return fetch('/preview', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data && data.preview) {
            previewData = data.preview.slice(0, 10); // Store first 5-10 rows for preview
            previewBtn.disabled = false; // Enable preview button after data is fetched
        } else {
            previewBtn.disabled = true;
            addMessageToChat('left', 'No preview available.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addMessageToChat('left', 'Error: Failed to fetch preview data.');
    });
}

// Toggle preview window when the "Show Table Preview" button is clicked
previewBtn.addEventListener('click', toggleTablePreview);

function toggleTablePreview() {
    if (!datasetUploaded || previewData.length === 0) {
        addMessageToChat('left', 'Please upload a dataset to preview.');
        return;
    }

    if (previewVisible) {
        hidePreviewWindow();
    } else {
        showPreviewWindow();
    }
}

function showPreviewWindow() {
    // Clean up any existing preview first
    cleanupExistingPreview();
    
    const previewWindow = document.createElement('div');
    previewWindow.classList.add('preview-window');
    previewWindow.innerHTML = generatePreviewTable(previewData);
    document.body.appendChild(previewWindow);
    previewVisible = true;
    previewBtn.textContent = "Hide Table Preview";
}

function hidePreviewWindow() {
    cleanupExistingPreview();
}

function generatePreviewTable(previewData) {
    let tableHtml = '<table class="preview-table"><thead><tr>';

    // Generate table headers
    const headers = Object.keys(previewData[0]);
    headers.forEach(header => {
        tableHtml += `<th>${header}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    // Generate table rows
    previewData.forEach(row => {
        tableHtml += '<tr>';
        headers.forEach(header => {
            tableHtml += `<td>${row[header]}</td>`;
        });
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table>';
    return tableHtml;
}

function addLoadingMessage() {
    // Create message structure similar to regular messages
    const loadingMessage = document.createElement('li');
    loadingMessage.classList.add('message', 'left', 'appeared');
    loadingMessage.id = 'loading-message';

    // Create the avatar (same as regular messages)
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');

    // Create text wrapper (same as regular messages)
    const textWrapper = document.createElement('div');
    textWrapper.classList.add('text_wrapper');

    // Create the actual loading content
    const loadingContent = document.createElement('div');
    loadingContent.classList.add('text');
    loadingContent.innerHTML = `
        <div class="loading-wrapper">
            <svg class="spinner" viewBox="0 0 24 24">
                <path d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z" fill="#4b5563"/>
            </svg>
            <span>Working on it... This may take a few seconds.</span>
        </div>
    `;

    // Assemble the message in the same order as regular messages
    textWrapper.appendChild(loadingContent);
    loadingMessage.appendChild(avatar);
    loadingMessage.appendChild(textWrapper);

    // Add to chat and scroll
    responseText.appendChild(loadingMessage);
    responseText.scrollTop = responseText.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMessage = document.getElementById('loading-message');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// Helper function to render different types of responses
async function renderResponseContent(responseData) {
    const contentWrapper = document.createElement('div');
    contentWrapper.classList.add('text_wrapper');

    try {
        // Handle visualization if present
        if (responseData.vegaSpec) {
            const vegaSpec = { ...responseData.vegaSpec };
            if (vegaSpec.data && vegaSpec.data.values === "myData") {
                vegaSpec.data.values = parsedData;
            }
            const chartElement = await renderChart(vegaSpec);
            contentWrapper.appendChild(chartElement);
        }

        // Handle text response if present (data analysis or error message)
        if (responseData.response) {
            const textElement = document.createElement('div');
            textElement.classList.add('text');

            // Parse and render the Markdown content
            textElement.innerHTML = marked.parse(responseData.response);

            contentWrapper.appendChild(textElement);
        }

        return contentWrapper;
    } catch (error) {
        console.error('Error rendering response:', error);
        const errorElement = document.createElement('div');
        errorElement.classList.add('text');
        errorElement.textContent = 'Error rendering response: ' + error.message;
        contentWrapper.appendChild(errorElement);
        return contentWrapper;
    }
}


async function sendMessage() {
    const userQuery = promptInput.value.trim();
    if (!userQuery) {
        return;
    }

    promptInput.value = '';
    sendBtn.disabled = true;
    addMessageToChat('right', userQuery);

    if (!datasetUploaded) {
        addMessageToChat('left', 'Please upload a dataset before sending a message.');
        sendBtn.disabled = false;
        return;
    }

    addLoadingMessage();

    try {
        // Prepare the request (keeping the current structure for now)
        const columns = getColumnNames(previewData);
        const types = getDataTypes(previewData);
        const sampleValues = getSampleValues(previewData);
        const vegaPrompt = generateVegaLitePrompt(columns, types, sampleValues, userQuery);

        const response = await fetch('/query', {
            method: 'POST',
            body: JSON.stringify({
                prompt: vegaPrompt,
                userQuery: userQuery
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error((await response.json()).detail || 'Server error');
        }

        const responseData = await response.json();
        removeLoadingMessage();

        // Create message container
        const newMessage = document.createElement('li');
        newMessage.classList.add('message', 'left', 'appeared');
        
        // Add avatar
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        newMessage.appendChild(avatar);

        // Render and add content
        const contentWrapper = await renderResponseContent(responseData);
        newMessage.appendChild(contentWrapper);
        
        // Add to chat and scroll
        responseText.appendChild(newMessage);
        responseText.scrollTop = responseText.scrollHeight;

    } catch (error) {
        removeLoadingMessage();
        console.error('Error:', error);
        addMessageToChat('left', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
    }
}


// Function to add a message to the chatbox
function addMessageToChat(side, text) {
    // Create the new message list item
    const newMessage = document.createElement('li');
    newMessage.classList.add('message', side, 'appeared');

    // Create the avatar
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');

    // Create the text wrapper
    const textWrapper = document.createElement('div');
    textWrapper.classList.add('text_wrapper');

    // Create the message text element
    const messageText = document.createElement('div');
    messageText.classList.add('text');

    // Render Markdown for assistant messages
    if (side === 'left') {
        messageText.innerHTML = marked.parse(text);
    } else {
        // For user messages, display as plain text to prevent XSS attacks
        messageText.textContent = text;
    }

    // Append the text inside the text wrapper
    textWrapper.appendChild(messageText);

    // Add the avatar and the text wrapper to the message
    newMessage.appendChild(avatar);
    newMessage.appendChild(textWrapper);

    // Append the new message to the chatbox
    responseText.appendChild(newMessage);

    // Scroll to the bottom of the chatbox after adding the message
    responseText.scrollTop = responseText.scrollHeight;
}


// Function to render a Vega-Lite chart in the chatbox
function renderChart(vegaSpec) {
    return new Promise((resolve, reject) => {
        console.log("Vega-Lite Specification:", JSON.stringify(vegaSpec, null, 2));
        const chartDiv = document.createElement('div');

        vegaEmbed(chartDiv, vegaSpec)
            .then(() => {
                resolve(chartDiv); // Return the chart element
            })
            .catch(error => {
                console.error('Error rendering Vega-Lite chart:', error);
                addMessageToChat('left', 'Unfortunately, the Vega-Lite specification is ill-formed, please send your request again.');
                reject(error);
            });
    });
}

function addCombinedMessageToChat(side, chartElement, descriptionText) {
    // Create the new message list item
    const newMessage = document.createElement('li');
    newMessage.classList.add('message', side, 'appeared');

    // Create the avatar
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');

    // Create the content wrapper
    const contentWrapper = document.createElement('div');
    contentWrapper.classList.add('text_wrapper');

    // Append the chart element to the content wrapper
    contentWrapper.appendChild(chartElement);

    // Create the description text element
    const messageText = document.createElement('div');
    messageText.classList.add('text');
    messageText.textContent = descriptionText;

    // Append the description text to the content wrapper
    contentWrapper.appendChild(messageText);

    // Add the avatar and the content wrapper to the message
    newMessage.appendChild(avatar);
    newMessage.appendChild(contentWrapper);

    // Append the new message to the chatbox
    responseText.appendChild(newMessage);

    // Scroll to the bottom of the chatbox after adding the message
    responseText.scrollTop = responseText.scrollHeight;
}


// Send the message when Enter is pressed
promptInput.addEventListener('keyup', function(event) {
    if (event.key === 'Enter') {
        sendBtn.click();
    }
});

// Send the message when the send button is clicked
sendBtn.addEventListener('click', sendMessage);