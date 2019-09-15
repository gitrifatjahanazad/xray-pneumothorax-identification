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
          if (res.status === 201)
            window.alert("File uploaded");
            var getCanvasImage = function(img) {
              var canvas = document.createElement('canvas');
              var context = canvas.getContext('2d');
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
              context.drawImage(img, 0, 0, img.naturalWidth,    img.naturalHeight,      // source rectangle
              0, 0, img.naturalWidth, img.naturalHeight);                               // destination rectangle
              
              return canvas;
            }
              
            
            
            
            setTimeout(() => {
              var originalImg = document.getElementById("originalImg");
              var resultImg = document.getElementById("resultImg");
              var originalCanvas = getCanvasImage(originalImg);
              var originalImgData = originalCanvas.getContext('2d').getImageData(0, 0, originalImg.naturalWidth, originalImg.naturalHeight);
              
              var resultCanvas = getCanvasImage(resultImg);
              var resultImgData = resultCanvas.getContext('2d').getImageData(0, 0, originalImg.naturalWidth, originalImg.naturalHeight);
                for(var i = 0; i< originalImgData.data.length;i+= 4){
                  if(resultImgData.data[i] !== 0 ||
                    resultImgData.data[i+1] !== 0 ||
                    resultImgData.data[i+2] !== 0){
                      originalImgData.data[i] = 255;
                  }
                //imageData.data[i+1] = imageData.data[i+1] ^ 255; // Invert Green
                //imageData.data[i+2] = imageData.data[i+2] ^ 255; // Invert Blue
                }
      
                originalCanvas.getContext('2d').putImageData(originalImgData, 0, 0);
                resultImg.src = originalCanvas.toDataURL();
            });
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
        <div className="imageSection">
          <div className="originalImgSec">
            <img id="originalImg" alt="original" />
            <label>Original Image</label>
          </div>
          <div className="resultImgSec">
            <img id="resultImg" alt="result" />
            <label>Pneumothorax Identified Region</label>
          </div>
        </div>
        <div className="fileUploadBtn">Upload Your X-Ray
          <input id="uploadInput" type="file" name="file" onChange={onChangeHandler} accept="image/x-png" />
        </div>
      </form>
    </div>
  );
}

export default App;
