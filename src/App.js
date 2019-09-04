import React from 'react';
// import logo from './logo.svg';
import './App.css';
import axios from 'axios';

function App() {
  let onChangeHandler = function (event) {
    console.log(event.target.files[0]);
    const fileReaderInstance = new FileReader();
    fileReaderInstance.readAsDataURL(event.target.files[0]);
    fileReaderInstance.onload = () => {
      let originalImg = fileReaderInstance.result;
      document.querySelector("#originalImg").src=originalImg;
      console.log(originalImg);
    }

    axios.post("https://testxray.free.beeceptor.com/file", event.target.files[0], {}).then(function (res) {
      console.log(res)
      if (res.status === 200)
        window.alert("File uploaded");
    });
  }

  return (
    <div className="App">
      {/* <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header> */}
      <form>
        <input id="uploadInput" type="file" name="file" onChange={onChangeHandler} accept="image/x-png" />
        <img id="originalImg" alt="original" />
      </form>
    </div>
  );
}

export default App;
