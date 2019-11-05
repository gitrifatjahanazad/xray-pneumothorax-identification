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
      
      <header class="d-flex justify-content-inline sub_menu">
        <div class="container">
          <div class="row" id="header">

       
            <div class="col-lg-2 col-2">
              <img class="img_style" src="apple-touch-icon1.png" alt="logo" class="logo"></img>
            </div>
            <div class="col-lg-10 col-2 my-4">

              <h1 class="heading_text">Upload X-Ray To Find Pneumothorax</h1>
            </div>
            
         
        </div>
        </div>
      </header>

{/*main*/}
  <main id="main_menu">  

<div class="container">
      <section class="row">
<div class="two col-lg-12 col-sm-2" id="services"> 
          <p class="h5 pt-3 texting1">What is pneumothorax(Collapsed lung)?</p>
          <h6 class="h6 texting2">"air on the wrong side of the lung", can result in death</h6>   
</div>
</section>
          <div class="row text-light">         
          <div class="three col-lg-3 col-sm-2  mt-3 text-left">
          <p class="sub_texting">
                        A pneumothorax is an abnormal collection of air in the pleural space between the lung and the
                        chest wall. Symptoms typically include sudden onset of sharp, one-sided chest pain and shortness
                        of breath.</p>
                        <p class="sub_texting">Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or
                            most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed
                            lung can be a life-threatening event. </p>
                        <p class="sub_texting">Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be
                            very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful
                            in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority
                            interpretation, or to provide a more confident diagnosis for non-radiologists.</p>
                    
                    <p class="h6 text-center sub_texting1">Check More On Wikipedia</p>
          </div>
         
<div class="one col-lg-6 col-sm-2 mt-3 px-4">
  <section class="row d-flex justify-content-end">
  <a class="nav-link" href="#services"><button class=" d-block d-lg-none d-sm-block btn btn-default btn-circle">&nbsp;Read <br></br>More</button></a>


</section>
            <section class="row justify-content-around">
         
              <img id="originalImg" alt="Original Image" src="x-ray.png"/>       
              <img id="originalImg" alt="Original Image" src="x-ray.png"/>   
                               
            </section>

            <p class="row justify-content-center mt-4">
              <div class="">Pneumothorax X-rays</div>
              </p>
       
             <section class="row justify-content-center mt-4 px-4">
                    <button class="btn btn_style">Upload Your X-Ray</button>
             </section>
      
       {/*<section class="fileUploadBtn row justify-content-center mt-3 px-4">
              Upload Your X-Ray
          <input id="uploadInput" type="file" name="file" onChange={onChangeHandler} accept="image/x-png" />
        </section>

    */}
</div>           

<div class="three col-lg-3 col-sm-2 mt-3 pl-4">

<p class="h6 texting3">About This Project</p>
<p class="pt-3 pl-4 texting5">This Project was developed<br></br>
                        to Research Purpose of <br></br>
                        Brain Station 23.<br></br>
                        A brain Child of kaggle <br></br>
                        "SIIM-ACR Pneumothorax <br></br>
                        Segmentation" Used Unet with <br></br>
                        inception resnet v2.</p>

                   
                    <p class="h6 pt-4 texting4" >Special Thanks To</p>

                    <p class="pt-3 pl-4 texting6">Raisul Kabir<br></br>
                        Jesper<br></br>
                        Siddhartha<br></br>
                        Ekhtiar Syed<br></br>
                        Rishabh Agrahari<br></br></p>
                        </div>

                      
<hr class="style1"></hr>

{/*footer */}
 
 <footer class="footer_style mt-3">
      <div class="row ">


      
<div class="col-lg-3 col-sm-2"></div>

<div class="five col-lg-2 col-sm-2">
<img src="kaggle3.png" alt="logo"></img>
</div>
<div class="six col-lg-2 col-sm-2">
<img src="Layer_1.svg" alt="logo1" class="bs_23"></img>
</div>

<div class="seven col-lg-2 col-sm-2">
<img src="buet.png" alt="logo"></img>
<p class="logo_text"><b>BUET</b></p>  
</div>

<div class="col-lg-3 col-sm-2"></div>
              
      </div>
        
 </footer>
 &nbsp;
 </div>
      </div>
      </main>
    </div>
    
  );
}

export default App;