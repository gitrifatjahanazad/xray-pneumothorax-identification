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
      document.querySelector("#originalImg").src = originalImg;

      document.querySelector("#resultImg").src = originalImg;
      document.querySelector("#resultImg").style.display = "block";
      console.log(originalImg);
      var formData = new FormData();
      var imagefile = document.querySelector('#uploadInput');
      formData.append("file", imagefile.files[0]);
      axios.post("http://127.0.0.1:8000", formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }).then(function (res) {
          console.log(res)
          let resImg = 'data:image/png;base64,' + res.data.response;
          document.querySelector("#resultImg").src = resImg;
          if (res.status === 200)
            window.alert("File uploaded");
        });
    }


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
        <div className="imageSection">
          <img id="originalImg" alt="original" />
          <img id="resultImg" alt="result" />
        </div>
      </form>
    </div>
  );
}

export default App;
