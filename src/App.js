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
      document.querySelector("#originalImg").style.display = "block";
      document.querySelector("#originalImg").src=originalImg;
      
      document.querySelector("#resultImg").src=originalImg;
      document.querySelector("#resultImg").style.display = "block";
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
      <header>
        <h1>Welcome</h1>
      </header>
      <form>
        <div className="fileUploadBtn">Upload Your X-Ray
          <input id="uploadInput" type="file" name="file" onChange={onChangeHandler} accept="image/x-png" />
        </div>
        <div class="imageSection">
        <img id="originalImg" alt="original" />
        <img id="resultImg" alt="result" />
        </div>
      </form>
    </div>
  );
}

export default App;
