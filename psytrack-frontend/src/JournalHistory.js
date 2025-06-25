import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './JournalHistory.css';

function JournalHistory() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Fetch entries on component mount
  useEffect(() => {
    const fetchEntries = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/entries');
        setEntries(response.data);
      } catch (err) {
        setError('Failed to load entries.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchEntries();
  }, []);

  if (loading) return <p>Loading your journal history…</p>;
  if (error)   return <p>{error}</p>;

  return (
    <div className="history-container">
      <h2>Your Journal History</h2>
      {entries.length === 0 ? (
        <p>No entries found.</p>
      ) : (
        <ul>
          {entries.map((e) => (
            <li key={e.id} className="history-entry">
              <p><strong>Mood:</strong> {e.mood}</p>
              <p><strong>Entry:</strong> {e.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default JournalHistory;
