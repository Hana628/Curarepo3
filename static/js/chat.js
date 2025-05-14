// Chat functionality for CURA health assistant
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const userMessageInput = document.getElementById('user-message');
    const sendMessageBtn = document.getElementById('send-message');
    const suggestionChipsContainer = document.getElementById('suggestion-chips');
    const loadingMessage = document.getElementById('loading-message');
    
    // Track typing status
    let botIsTyping = false;
    
    // Check if we're in the chat section - only initialize if so
    if (chatMessages) {
        console.log("Chat interface detected, initializing...");
        initializeChat();
    } else {
        console.warn("Chat interface elements not found, chat initialization skipped");
        return; // Exit if we're not on a page with the chat interface
    }
    
    // Set up event listeners for sending messages
    if (sendMessageBtn && userMessageInput) {
        // Send message on button click
        sendMessageBtn.addEventListener('click', function() {
            sendUserMessage();
        });
        
        // Send message on Enter key press
        userMessageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendUserMessage();
            }
        });
    } else {
        console.error("Chat input elements not found, chat functionality may be limited");
    }
    
    /**
     * Initialize the chat interface and fetch the welcome message
     */
    function initializeChat() {
        console.log("Fetching welcome message...");
        
        // Use the existing loading message or create one if needed
        const existingLoadingMessage = document.getElementById('loading-message');
        
        // Fetch the welcome message from the server
        fetch('/get_welcome_message')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Welcome message received:", data);
                
                // Remove loading message if it exists
                if (existingLoadingMessage && existingLoadingMessage.parentNode) {
                    existingLoadingMessage.parentNode.removeChild(existingLoadingMessage);
                }
                
                if (data.status === 'success') {
                    // Display welcome message
                    displayBotMessage(data.message);
                } else {
                    // Display error message
                    displayBotMessage({
                        text: "I'm having trouble connecting to my knowledge base. Please try again later."
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching welcome message:', error);
                
                // Remove loading message if it exists
                if (existingLoadingMessage && existingLoadingMessage.parentNode) {
                    existingLoadingMessage.parentNode.removeChild(existingLoadingMessage);
                }
                
                // Display error message
                displayBotMessage({
                    text: "I'm having trouble connecting to my knowledge base. Please try again later."
                });
            });
    }
    
    /**
     * Send the user's message to the server and handle the response
     */
    function sendUserMessage() {
        // Validate input and check if bot is already responding
        if (!userMessageInput || !userMessageInput.value.trim() || botIsTyping) return;
        
        const message = userMessageInput.value.trim();
        userMessageInput.value = ''; // Clear input field
        
        // Display user message in the chat
        displayUserMessage(message);
        
        // Show typing indicator
        showTypingIndicator();
        
        console.log("Sending message to server:", message);
        
        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Server response:", data);
            
            // Hide typing indicator
            hideTypingIndicator();
            
            if (data.status === 'success') {
                // Log full response object for debugging
                console.log("Full response object:", data.response);
                
                // Display bot response
                displayBotMessage(data.response);
                
                // Handle any actions in the response
                if (data.response && data.response.action) {
                    handleBotAction(data.response.action);
                }
            } else {
                // Display error message
                displayBotMessage({
                    text: "I'm sorry, I encountered an error. Please try again."
                });
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Display error message
            displayBotMessage({
                text: "I'm having trouble connecting. Please try again later."
            });
        });
    }
    
    /**
     * Display a user message in the chat
     */
    function displayUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;
        
        // Add timestamp
        const timeElement = document.createElement('div');
        timeElement.className = 'message-time';
        timeElement.textContent = getCurrentTime();
        messageElement.appendChild(timeElement);
        
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    /**
     * Display a bot message in the chat
     */
    function displayBotMessage(response) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        
        console.log("Displaying bot message with response:", response);
        
        // Add text content with markdown formatting
        if (response && response.text) {
            messageElement.innerHTML = formatMessageText(response.text);
        } else if (typeof response === 'string') {
            // Handle case where response is just a string
            messageElement.innerHTML = formatMessageText(response);
        } else {
            messageElement.textContent = 'I apologize, but I got an empty response. How else can I help you?';
        }
        
        // Add suggestion options if present
        if (response && response.options && response.options.length > 0) {
            const optionsElement = document.createElement('div');
            optionsElement.className = 'message-options';
            
            response.options.forEach(option => {
                const optionElement = document.createElement('div');
                optionElement.className = 'message-option';
                optionElement.textContent = option;
                
                optionElement.addEventListener('click', function() {
                    userMessageInput.value = option;
                    sendUserMessage();
                });
                
                optionsElement.appendChild(optionElement);
            });
            
            messageElement.appendChild(optionsElement);
        }
        
        // Add timestamp
        const timeElement = document.createElement('div');
        timeElement.className = 'message-time';
        timeElement.textContent = getCurrentTime();
        messageElement.appendChild(timeElement);
        
        chatMessages.appendChild(messageElement);
        scrollToBottom();
        
        // Update suggestion chips
        if (response && (response.options || (typeof response === 'object'))) {
            updateSuggestionChips(response);
        }
    }
    
    /**
     * Format message text with basic markdown support
     */
    function formatMessageText(text) {
        if (!text) return '';
        
        // Handle headers (## Header)
        text = text.replace(/## (.*?)(\n|$)/g, '<h5>$1</h5>');
        
        // Handle bold (**text**)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Handle italic (*text*)
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Handle bullet lists
        text = text.replace(/^\* (.*?)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*?<\/li>(\n|$))+/g, '<ul>$&</ul>');
        
        // Convert line breaks to HTML breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    /**
     * Update suggestion chips below the chat input
     */
    function updateSuggestionChips(response) {
        if (!suggestionChipsContainer) return;
        
        // Clear existing chips
        suggestionChipsContainer.innerHTML = '';
        
        // Add new chips if options are present
        if (response && response.options && Array.isArray(response.options) && response.options.length > 0) {
            response.options.forEach(option => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.textContent = option;
                
                chip.addEventListener('click', function() {
                    userMessageInput.value = option;
                    sendUserMessage();
                });
                
                suggestionChipsContainer.appendChild(chip);
            });
            
            // Make sure the suggestion chips container is visible
            suggestionChipsContainer.style.display = 'flex';
        } else {
            // Hide suggestion chips container if no options
            suggestionChipsContainer.style.display = 'none';
        }
    }
    
    /**
     * Show the bot typing indicator
     */
    function showTypingIndicator() {
        botIsTyping = true;
        
        const typingElement = document.createElement('div');
        typingElement.className = 'typing-indicator';
        typingElement.id = 'typing-indicator';
        typingElement.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        
        chatMessages.appendChild(typingElement);
        scrollToBottom();
    }
    
    /**
     * Hide the bot typing indicator
     */
    function hideTypingIndicator() {
        botIsTyping = false;
        
        const typingElement = document.getElementById('typing-indicator');
        if (typingElement && typingElement.parentNode === chatMessages) {
            chatMessages.removeChild(typingElement);
        }
    }
    
    /**
     * Handle bot actions to show different sections
     */
    function handleBotAction(action) {
        console.log("Handling bot action:", action);
        
        try {
            // Hide all sections
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.add('d-none');
            });
            
            // Show the appropriate section based on action
            if (action === 'showBloodPressureForm') {
                const section = document.getElementById('blood-pressure-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-symptoms');
                }
            } else if (action === 'showDiabetesForm') {
                const section = document.getElementById('diabetes-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-symptoms');
                }
            } else if (action === 'showLifestyleForm') {
                const section = document.getElementById('lifestyle-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-health');
                }
            } else if (action === 'showAnomalyForm') {
                const section = document.getElementById('anomaly-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-symptoms');
                }
            } else if (action === 'showPredictionOptions') {
                const section = document.getElementById('symptoms-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-symptoms');
                }
            } else if (action === 'showAppointmentForm') {
                const section = document.getElementById('appointments-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-appointments');
                }
            } else if (action === 'showSkinDiseaseForm') {
                const section = document.getElementById('skin-disease-section');
                if (section) {
                    section.classList.remove('d-none');
                    activateNavTab('show-symptoms');
                }
            }
        } catch (error) {
            console.error("Error handling bot action:", error);
        }
    }
    
    /**
     * Activate a navigation tab programmatically
     */
    function activateNavTab(action) {
        try {
            // First try to find nav link in the sidebar
            const navLinks = document.querySelectorAll('.nav-link');
            
            // Remove active class from all nav links
            navLinks.forEach(link => {
                link.classList.remove('active');
            });
            
            // Find and activate the correct tab
            const navLink = Array.from(navLinks).find(link => link.getAttribute('data-action') === action);
            if (navLink) {
                navLink.classList.add('active');
            }
        } catch (error) {
            console.error("Error activating nav tab:", error);
        }
    }
    
    /**
     * Scroll the chat messages to the bottom
     */
    function scrollToBottom() {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    /**
     * Get the current time formatted as HH:MM AM/PM
     */
    function getCurrentTime() {
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        
        hours = hours % 12;
        hours = hours ? hours : 12; // Hour '0' should be '12'
        minutes = minutes < 10 ? '0' + minutes : minutes;
        
        return `${hours}:${minutes} ${ampm}`;
    }
});
