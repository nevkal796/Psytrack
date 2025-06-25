import React, { useState } from 'react';
import axios from 'axios';
import './JournalEntry.css';

function JournalEntry() {
  const [text, setText] = useState('');
  const [mood, setMood] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://127.0.0.1:8000/entry', { text, mood });
      setMessage('Entry submitted successfully!');
      setText('');
      setMood('');
    } catch (error) {
      setMessage('Failed to submit entry.');
      console.error(error);
    }
  };

  return (
    <div className="journal-container">
      <h2>New Journal Entry</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Write how you're feeling..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows="5"
          cols="40"
        />
        <br />
        <select value={mood} onChange={(e) => setMood(e.target.value)}>
          <option value="">Select mood</option>
          <option value="happy">😊 Happy</option>
          <option value="sad">😢 Sad</option>
          <option value="anxious">😰 Anxious</option>
          <option value="angry">😠 Angry</option>
        </select>
        <br /><br />
        <button type="submit">Submit Entry</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
}

export default JournalEntry;
