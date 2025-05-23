function sendMessage() {
    let messageInput = document.getElementById('message-input');
    let message = messageInput.value;
    displayMessage('user', message);
    
    // Get the selected function from the dropdown menu
    let functionSelect = document.getElementById('function-select');
    let selectedFunction = functionSelect.value;
    
    // Send an AJAX request to the Flask API endpoint based on the selected function
    let xhr = new XMLHttpRequest();
    let url;

    switch (selectedFunction) {
        case 'search':
            url = '/search';
            break;
        case 'kbanswer':
            url = '/kbanswer';
            break;
        case 'answer':
            url = '/answer';
            break;
        default:
            url = '/answer';
    }
    
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            displayMessage('assistant', response.message);
        }
    };
    xhr.send(JSON.stringify({message: message}));
    
    // Clear the input field
    messageInput.value = '';
}

// Function to display messages
function displayMessage(sender, message) {
    let chatContainer = document.getElementById('chat-container');
    let messageDiv = document.createElement('div');

    if (sender === 'assistant') {
        messageDiv.classList.add('assistant-message');
    } else {
        messageDiv.classList.add('user-message');
    }

    messageDiv.innerHTML = `<b>${sender.charAt(0).toUpperCase() + sender.slice(1)}:</b> ${message}`;

    // Add a timestamp
    let timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    timestamp.innerText = " [" + new Date().toLocaleTimeString() + "]";
    messageDiv.appendChild(timestamp);

    chatContainer.appendChild(messageDiv);

    // Auto-scroll to the bottom of the chat container
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle button click event
let sendButton = document.getElementById('send-btn');
sendButton.addEventListener('click', sendMessage);

// Clear chat history when clicked
let clearButton = document.getElementById('clear-chat-btn');
clearButton.addEventListener('click', function() {
    document.getElementById('chat-container').innerHTML = '';
});
