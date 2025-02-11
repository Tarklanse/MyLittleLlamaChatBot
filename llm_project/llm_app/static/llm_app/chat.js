document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const clearButton = document.getElementById('clearButton');
    const undoButton = document.getElementById('undoButton');
    const retryButton = document.getElementById('retryButton'); 
    const historyList = document.getElementById('historyList');
    const ttsSwitch = document.getElementById('ttsSwitch');
    const newChatButton = document.getElementById('newChatButton');
    
    const editPersonalButton = document.getElementById('editPersonalButton');
    const personalEditModal = document.getElementById('personalEditModal');
    const closeModalButton = document.getElementById('closeModalButton');
    const savePersonalButton = document.getElementById('savePersonalButton');
    const personalInput = document.getElementById('personalInput');
    const personalDiv = document.getElementById('Personal');
    // Check for browser support for the Web Speech API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        // Configure speech recognition
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'zh-TW'; // Set language (you can change this)

        // Trigger speech recognition when microphone button is clicked
        micButton.addEventListener('click', () => {
            recognition.start(); // Start speech recognition
        });

        // Handle results from speech recognition
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript; // Insert recognized speech into text input
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };
    } else {
        console.warn('Speech Recognition API is not supported in this browser.');
    }
    // Fetch chat history and display it in the sidebar
    async function loadChatHistory() {
        try {
            const response = await fetch('/chat/gethistory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin'  // To include session cookies
            });
            
            const data = await response.json();
            if (response.ok) {
                displayChatHistory(data.result);
            } else if (data.redirect_url) {
                window.location.href = data.redirect_url;
            } else {
                console.error("Failed to load chat history.");
            }
        } catch (error) {
            console.error("Error loading chat history:", error);
        }
    }
    function checkHistory() {
        let ccidvalue = document.getElementById('CCID').getAttribute('data-value');
        if (ccidvalue != ''){
            if (!isCCIDInHistory(ccidvalue)){
                loadChatHistory();
            }
        }
    }
    function getHistoryItemsList() {
        const historyItems = document.querySelectorAll('.history-item');
        const historyList = Array.from(historyItems).map(item => item.getAttribute('data-ccid'));
        return historyList;
    }
    
    // Example usage to check if a specific `CCID` is in the list
    function isCCIDInHistory(ccidToCheck) {
        const historyList = getHistoryItemsList();
        return historyList.includes(ccidToCheck);
    }
    // Populate the chat history in the sidebar
    function displayChatHistory(historyListData) {
        historyList.innerHTML = '';  // Clear existing items
        historyListData.forEach(session => {
            const listItem = document.createElement('li');
            listItem.textContent = session;  // Add session details here
            listItem.classList.add('history-item');
            historyList.appendChild(listItem);
            
            // Add click listener to load this specific session if needed
            console.log(session)
            listItem.addEventListener('click', () => {
                loadChatSession(session); // You may implement loadChatSession to retrieve and display messages of this session
            });
        });
    }

    // Call loadChatHistory when the page loads
    loadChatHistory();
    // Function to show the blocker
    function showBlocker() {
        document.getElementById("blocker").style.display = "flex";
    }

    // Function to hide the blocker
    function hideBlocker() {
        document.getElementById("blocker").style.display = "none";
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value;
        if (message.trim() === '') return;
        showBlocker()
        addMessageToChat('You', message);
        userInput.value = '';
        let value = document.getElementById('CCID').getAttribute('data-value');
        // Send message to LLM API
        const response = await fetch('/api/llm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // Include CSRF token if necessary
            },
            credentials: 'same-origin',  // Ensures the session ID cookie is sent with the request
            body: JSON.stringify({ prompt: message ,CCID : value})
        });
        
        // Check if response status is 403 (redirect needed)
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;  // Redirect to login page
            return;
        }
        // Check if the response contains a user_id and display it in the chat
        
        const data = await response.json();
        console.log(data.status)
        if (response.ok) {
            addMessageToChat('Bot', data.result);
            document.getElementById('CCID').setAttribute('data-value', data.CCID );
                // Check if TTS is enabled
            if (ttsSwitch.checked) {
                speakText(data.result); // Call the TTS function if enabled
            }
        } else {
            addMessageToChat('Bot', 'Error: ' + data.message);
        }
        hideBlocker()
        checkHistory()
    });

    clearButton.addEventListener('click', async () => {
        let value = document.getElementById('CCID').getAttribute('data-value');
        const chatMessages = chatBox.getElementsByClassName('chat-message');
        if (chatMessages.length==0 | value==''){
            return
        }
        // Call the retry API
        const response = await fetch('/chat/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
            },
            credentials: 'same-origin',  // Ensures session ID cookie is sent
            body: JSON.stringify({ CCID : value})
        });
        
        // Check if response status is 403 (redirect needed)
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;  // Redirect to login page
            return;
        }
        chatBox.innerHTML = '';  // Clear chat box
    });
    newChatButton.addEventListener('click', async () => {
        document.getElementById('CCID').setAttribute('data-value', '');
        chatBox.innerHTML = '';  // Clear chat box
    });
    // Retry button functionality
    undoButton.addEventListener('click', async () => {
        
        let value = document.getElementById('CCID').getAttribute('data-value');
        const chatMessages = chatBox.getElementsByClassName('chat-message');
        if (chatMessages.length==0 | value==''){
            return
        }
        // Call the retry API
        const response = await fetch('/chat/undo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
            },
            credentials: 'same-origin',  // Ensures session ID cookie is sent
            body: JSON.stringify({ CCID : value})
        });

        // Check if response status is 403 (redirect needed)
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;  // Redirect to login page
            return;
        }
        if (response.status === 304) {
            return;
        }
        removeLastMessage();
        removeLastMessage();
    });
    // Retry button functionality
    retryButton.addEventListener('click', async () => {
        
        let value = document.getElementById('CCID').getAttribute('data-value');
        const chatMessages = chatBox.getElementsByClassName('chat-message');
        // Check if there are any messages and if the last one is from the Bot
        if (chatMessages.length > 0) {
            const lastMessage = chatMessages[chatMessages.length - 1];
            if (lastMessage.innerHTML.startsWith('<strong>Bot:</strong>')) {
                removeLastMessage(); // Remove only if the last message is from the Bot
            }
        }

        // Ensure there's still a valid value and at least one message left
        if (chatMessages.length == 0 || value == '') {
            return;
        }

        // Repeat the check and remove the second-to-last message if it's also from the Bot
        if (chatMessages.length > 0) {
            const lastMessage = chatMessages[chatMessages.length - 1];
            if (lastMessage.innerHTML.startsWith('<strong>Bot:</strong>')) {
                removeLastMessage();
            }
        }
        showBlocker()
        // Call the retry API
        const response = await fetch('/chat/retry', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
            },
            credentials: 'same-origin',  // Ensures session ID cookie is sent
            body: JSON.stringify({ CCID : value}),
        });
        // Check if response status is 403 (redirect needed)
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;  // Redirect to login page
            return;
        }
        const data = await response.json();
        if (response.ok) {
            addMessageToChat('Bot', data.result);
            // Check if TTS is enabled
            if (ttsSwitch.checked) {
                speakText(data.result); // Call the TTS function if enabled
            }
        } else {
            addMessageToChat('Bot', 'Error: Could not retry message.');
        }
        hideBlocker()
    });
    function addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
    
        // Convert markdown to HTML
        const htmlContent = marked.parse(message); // Parse markdown syntax to HTML
    
        // Create a temporary container to manipulate the DOM
        const tempContainer = document.createElement('div');
        tempContainer.innerHTML = htmlContent;
    
        // Find all <a> tags and update their href attributes
        const links = tempContainer.querySelectorAll('a');
        links.forEach(link => {
            if (link.href.startsWith('sandbox:')) {
                link.href = link.href.replace('sandbox:', '');
            }
        });
    
        // Set the innerHTML to the processed HTML content
        messageElement.innerHTML = `<strong>${sender}:</strong> ${tempContainer.innerHTML}`;
        
        const chatBox = document.getElementById('chatBox');
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
    }

    async function loadChatSession(CCID) {
        // Fetch chat session data from the server
        const response = await fetch('/chat/loadchat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
            },
            credentials: 'same-origin',
            body: JSON.stringify({ 'CCID': CCID })
        });
    
        // Redirect to login if 403 status
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;
            return;
        }
    
        const data = await response.json();
        
        // Clear the chatBox to display the loaded session
        chatBox.innerHTML = '';  
        
        // Iterate over the chat messages, skipping the first element (system message)
        data.result.slice(0).forEach(message => {
            if (message.role !== 'system') { // Skip processing when role is 'system'
                addMessageToChat(message.role === 'user' ? 'You' : 'Bot', message.content);
            }
            else {
                document.getElementById('Personal').setAttribute('value',message.content)
            }
        });
        
        document.getElementById('CCID').setAttribute('data-value', CCID );
    }

    function removeLastMessage() {
        const chatMessages = chatBox.getElementsByClassName('chat-message');
        // Find the last bot message
        const message = chatMessages[chatMessages.length - 1];
        message.remove();
    }
    // TTS function using the Web Speech API
    function speakText(text) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US'; // You can change the language code based on your needs
        window.speechSynthesis.speak(utterance);
    }
    // Helper function to get CSRF token from cookies
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    document.getElementById('uploadButton').addEventListener('click', async () => {
        const pdfInput = document.getElementById('pdfInput');
        const files = pdfInput.files;
        let value = document.getElementById('CCID').getAttribute('data-value');
        if (files.length === 0) {
            alert("Please select at least one PDF file to upload.");
            return;
        }
    
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('pdf_files', files[i]);
        }
        formData.append('CCID',value )
        try {
            const response = await fetch('/chat/upload_pdf', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
                },
                credentials: 'same-origin',
            });
            
            const data = await response.json();
            document.getElementById('CCID').setAttribute('data-value', data.CCID );
            if (response.ok) {
                alert('PDF(s) uploaded successfully!');
            } else {
                alert('Failed to upload PDF(s): ' + data.error);
            }
        } catch (error) {
            console.error('Error uploading PDF(s):', error);
        }
    });


    // Show modal when "Edit Personal" button is clicked
    editPersonalButton.addEventListener('click', () => {
        // Pre-fill the input with the current personal value
        const currentPersonal = personalDiv.getAttribute('value') || '';
        personalInput.value = currentPersonal;
        personalEditModal.style.display = 'block';
    });

    // Close modal when 'X' button is clicked
    closeModalButton.addEventListener('click', () => {
        personalEditModal.style.display = 'none';
    });

    // Save the edited personal value
    savePersonalButton.addEventListener('click', async () =>  {
        const newPersonalValue = personalInput.value.trim();
        let value = document.getElementById('CCID').getAttribute('data-value');
        const response = await fetch('/chat/editPersonal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
            },
            credentials: 'same-origin',
            body: JSON.stringify({ 'CCID': value ,'Personal':newPersonalValue})
        });
    
        // Redirect to login if 403 status
        if (response.status === 403) {
            const data = await response.json();
            window.location.href = data.redirect_url;
            return;
        }
        const data = await response.json();
        
        // Update the 'Personal' div's value attribute
        personalDiv.setAttribute('value', newPersonalValue);
        if (value!=data.CCID){
            document.getElementById('CCID').setAttribute('data-value', data.CCID );
        }
        // Optionally display confirmation
        alert('Personal information updated!');
        // Close the modal
        personalEditModal.style.display = 'none';
    });

    // Close modal when user clicks outside the modal content
    window.addEventListener('click', (event) => {
        if (event.target === personalEditModal) {
            personalEditModal.style.display = 'none';
        }
    });
});
