import React from 'react';
import JournalEntry from './JournalEntry';
import JournalHistory from './JournalHistory';

function App() {
  return (
    <div>
      <h1>Psytrack 🧠</h1>
      <JournalEntry />
      <hr />
      <JournalHistory />
    </div>
  );
}

export default App;
