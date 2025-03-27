import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import the CSS file

const API_URL = "http://127.0.0.1:8000";

const App = () => {
  const [sessionId, setSessionId] = useState("");
  const [chat, setChat] = useState([]);
  const [message, setMessage] = useState("");

  const addMessage = (role, text) => {
    setChat((prevChat) => [...prevChat, { role, text }]);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    addMessage("user", message);

    try {
      const response = await axios.get(`${API_URL}/recommend`, {
        params: { query: message, session_id: sessionId || undefined },
      });

      setSessionId(response.data.session_id);
      addMessage("bot", response.data.response);
      setMessage("");
    } catch (error) {
      console.error("Error sending message:", error);
      addMessage("bot", "Something went wrong. Please try again.");
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-title">Travel Chatbot</div>

      <div className="chat-area">
        {chat.map((msg, index) => (
          <div key={index} className={`message ${msg.role === "bot" ? "bot-message" : "user-message"}`}>
            <strong>{msg.role === "bot" ? "Bot: " : "You: "}</strong> {msg.text}
          </div>
        ))}
      </div>

      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about flights..."
          className="user-input"
        />
        <button onClick={sendMessage} className="send-button">Send</button>
      </div>
    </div>
  );
};

export default App;
