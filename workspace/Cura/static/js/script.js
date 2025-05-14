document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const loader = document.getElementById('loader');

    // Initialize the chat
    function initChat() {
        // Check if chat history exists in local storage
        const chatHistory = localStorage.getItem('cura-chat-history');
        if (chatHistory) {
            chatMessages.innerHTML = chatHistory;
            // Scroll to bottom of chat
            scrollToBottom();
        }
    }

    // Add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process message content for markdown-like formatting
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic text
            .replace(/\n/g, '<br>');                           // Line breaks
        
        messageContent.innerHTML = `<p>${formattedMessage}</p>`;
        messageDiv.appendChild(messageContent);
        
        chatMessages.appendChild(messageDiv);
        
        // Save chat history to local storage
        localStorage.setItem('cura-chat-history', chatMessages.innerHTML);
        
        // Scroll to the bottom of the chat
        scrollToBottom();
    }

    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Show loader
    function showLoader() {
        loader.classList.remove('d-none');
    }

    // Hide loader
    function hideLoader() {
        loader.classList.add('d-none');
    }

    // Send message to API
    async function sendMessage(message) {
        try {
            showLoader();
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'An error occurred while sending your message.');
            }
            
            const data = await response.json();
            hideLoader();
            
            // Add assistant's response to chat
            addMessage(data.response);
            
        } catch (error) {
            hideLoader();
            console.error('Error:', error);
            addMessage('Sorry, I encountered an error processing your request. Please try again later.');
        }
    }

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input field
        userInput.value = '';
        
        // Send message to API
        sendMessage(message);
    });

    // Initialize chat on page load
    initChat();
});
