import React, {useState} from 'react';
import axios from 'axios';
import './App.css';

function App() {  
  const [podName, setPodaName] = useState('');
  const [status, setStatus] = useState('');

  const spawnPod = () => {
    setStatus('Creating pod...');
    axios.post(`http://localhost:8080/api/spawn-pod?podName=${podName}`)
    .then(response => {
      setStatus(response.data);
    })
    .catch(error => {
      setStatus(`Error: ${error.message}`);
    });
  };

  return (
    <div className="App">
      <h1>Pod Creator</h1>
      <input 
        type="text" 
        value={podName} 
        onChange={(e) => setPodaName(e.target.value)} 
        placeholder="Enter pod name" 
      />
      <button onClick={spawnPod}>Create Pod</button>
      <p>{status}</p>
    </div>
  )
}

export default App;