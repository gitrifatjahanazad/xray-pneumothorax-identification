import React from 'react';
// import logo from './logo.svg';
import './App.css';
import axios from 'axios';

function App() {
  let onChangeHandler = function (event) {
    document.querySelector(".loader").style.display = "block";
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
          // if (res.status === 201)
          //   window.alert("File uploaded");
          var getCanvasImage = function (img) {
            var canvas = document.createElement('canvas');
            var context = canvas.getContext('2d');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            context.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight,      // source rectangle
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
            for (var i = 0; i < originalImgData.data.length; i += 4) {
              if (resultImgData.data[i] !== 0 ||
                resultImgData.data[i + 1] !== 0 ||
                resultImgData.data[i + 2] !== 0) {
                originalImgData.data[i] = 255;
              }
              //imageData.data[i+1] = imageData.data[i+1] ^ 255; // Invert Green
              //imageData.data[i+2] = imageData.data[i+2] ^ 255; // Invert Blue
            }

            originalCanvas.getContext('2d').putImageData(originalImgData, 0, 0);
            resultImg.src = originalCanvas.toDataURL();
            document.querySelector(".loader").style.display = "none";
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
      
      <header class="sub_menu">
        <div class="container">
          <div class="row">
            <div class="col-lg-1 ">
              <img class="img_style" src="apple-touch-icon1.png" alt="logo" class="logo"></img>
            </div>
            <div class="col-lg-11">
              <h1 class="head_title">Upload X-Ray To Find Pneumothorax</h1>
            </div>
           
          </div>
        </div>
      </header>

{/*main*/}

      <div class="main_bar">
        <div className="imageSection_main">

<div class="container">
          <p class="sub_title">What is pneumothorax(Collapsed lung)?</p>
          <h6 class="sub_title1">"air on the wrong side of the lung", can result in death</h6>

          <div class="row">
          <div class="col-md-3 order-2 order-md-1 section_1 featured_main">
            <p class="text-field">
              <div>
              <span class="text">A pneumothorax is an abnormal collection of air in the pleural space between the lung and the chest wall. Symptoms typically include sudden onset of sharp, one-sided chest pain and shortness of breath.</span>
            </div>
            <br></br>
            <div>
           <span class="text">Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event.</span> 
           </div>
           <br></br>
           <span class="text">Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios.
AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.</span>
</p>
            <p class="text-field1"><span class="text_style">Check More On Wikipedia</span></p>
          </div>
         
<div class="col-md-6 image_sub">
             
            <div className="Image_Section">
              <img id="originalImg" alt="Original Image" src="x-ray.png"/>
              <label> </label> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              &nbsp;&nbsp;&nbsp;
             
              <img id="originalImg" alt="Original Image" src="x-ray.png"/>
              <label></label>                         
            </div>
<div class="text_area">
            <p class="texting">Pneumothorax X-rays</p>
        </div>
        <div className="fileUploadBtn">
              Upload Your X-Ray
          <input id="uploadInput" type="file" name="file" onChange={onChangeHandler} accept="image/x-png" />
        </div>
</div>           
<br></br>
<div class="col-md-3 footer_area featured_sub">
<p class="text-field2">
          <div class="heading_sec">
            About This Project
            </div>
            <br></br>
            <span class="text">
            &nbsp;&nbsp;&nbsp;&nbsp;This Project was developed 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;to Research Purpose of 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;Brain Station 23.
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;A brain Child of kaggle 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;"SIIM-ACR Pneumothorax 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;Segmentation" 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;Used Unet with 
              <br></br> &nbsp;&nbsp;&nbsp;&nbsp;inception resnet v2.
          </span>
        </p>
        
        <br></br>
        <p class="text-field">
        <div class="heading_sec">
        Special Thanks To
            </div>
            <br></br>
             <span>
             &nbsp;&nbsp;&nbsp;&nbsp;Raisul Kabir 
              <br></br>  &nbsp;&nbsp;&nbsp;&nbsp;Jesper
              <br></br>  &nbsp;&nbsp;&nbsp;&nbsp;Siddhartha
              <br></br>  &nbsp;&nbsp;&nbsp;&nbsp;Ekhtiar Syed
              <br></br>  &nbsp;&nbsp;&nbsp;&nbsp;Rishabh Agrahari
             
            </span>
          </p>
</div>
</div>
</div>
<hr></hr>

{/*footer */}
 
    <div class="container section_style_logo">
      <div class="row">
      
      <div class="col-md-2">          
      <img src="kaggle3.png" alt="logo"></img>
          </div>
          <div class="col-md-4">
          <img src="Layer_1.svg" alt="logo1" class="bs_23"></img>
          </div>
          <div class="col-md-4">
          <img src="buet.png" alt="logo"></img>
            <h5 class="logo_text">BUET</h5>  
          </div>
          <div class="col-md-2"></div>
                                
      </div>
    </div>
 

      </div>
      </div>
    </div>
  );
}

export default App;

  