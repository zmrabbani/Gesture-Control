document.getElementById("chatbot_toggle").onclick = function () {
  if (document.getElementById("chatbot").classList.contains("collapsed")) {
    document.getElementById("chatbot").classList.remove("collapsed")
    document.getElementById("chatbot_toggle").children[0].style.display = "none"
    document.getElementById("chatbot_toggle").children[1].style.display = ""
    document.getElementById("chatbot_toggle").style.width = "48px"
    document.getElementById("close").style.display = "block"
  }
  else {
    document.getElementById("chatbot").classList.add("collapsed")
    document.getElementById("chatbot_toggle").children[0].style.display = ""
    document.getElementById("chatbot_toggle").children[1].style.display = "none"
    document.getElementById("chatbot_toggle").style.width = "125px"
  }
}



// teks
const msgerForm = get(".input-div");
const msgerInput = get(".input-message");

// chatroom
const msgerChat = get(".msger-chat");
const BOT_IMG = "../static/img/Logo/Logo2.png";
const PERSON_IMG = "../static/img/man.png";
const BOT_NAME = "SAVIBot";
const PERSON_NAME = "Kamu";


msgerForm.addEventListener("submit", event => {
  event.preventDefault();
  const msgText = msgerInput.value;
  if (!msgText) return;
  appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
  msgerInput.value = "";
  botResponse(msgText);
  map.remove();
});

function appendMessage(name, img, side, text) {
  //   Simple solution for small apps
  const msgHTML = `
<div class="msg ${side}-msg">
  <div class="msg-img" style="background-image: url(${img})"></div>
  <div class="msg-bubble">
    <div class="msg-info">
      <div class="msg-info-name">${name}</div>
      <div class="msg-info-time">${formatDate(new Date())}</div>
    </div>
    <div class="msg-text">${text}</div> 
  </div>
</div>
`;
  //msgerChat.innerHTML=msgHTML
  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop +=  500;
}

function appendMap(name, img, side) {
  //   Simple solution for small apps
  const msgHTML = `
<div class="msg ${side}-msg">
  <div class="msg-img" style="background-image: url(${img})"></div>
  <div class="msg-bubble">
    <div class="msg-info">
      <div class="msg-info-name">${name}</div>
      <div class="msg-info-time">${formatDate(new Date())}</div>
    </div>
    <div id="map"></div>
  </div>
</div>
`;
  //msgerChat.innerHTML=msgHTML
  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  savimap(-6.923987, 107.773354);
  msgerChat.scrollTop +=  500;
}

function savimap(lat,long){
  var iconsavi = L.icon({
    iconUrl: '../static/img/Logo/Logo2.png',
    iconSize:     [15, 25],
  });

  var map = L.map('map').setView([lat, long], 18);
  var marker = L.marker([lat, long], {icon: iconsavi}).addTo(map)
  .bindPopup('SAVI Disini!!');;

  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
  }).addTo(map);

  map.on('click', function(e) {
    var popLocation= e.latlng;
    console.log("Lat, Lon : " + e.latlng.lat + ", " + e.latlng.lng) 
    var popup = L.popup()
    .setLatLng(popLocation)
    .setContent("Lat : " + e.latlng.lat + "<br>" + "Lon : " + e.latlng.lng)
    .openOn(map);        
});
  
}

//savimap(-6.923987, 107.773354);


function botResponse(rawText) {
  // Bot Response
  $.get("/get", { prediction_input: rawText }).done(function (msg) {
    console.log(rawText);
    console.log(msg);
    const msgText = msg;
    appendMessage(BOT_NAME, BOT_IMG, "left", msgText);

    fetch('/tag').then(response => response.json()).then(data => {
      var label = data.response_tag;
      console.log(label)
      if (label == "SAVI.lokasi" ) {
        appendMap(BOT_NAME, BOT_IMG, "left");
      }
    });
    
  });
  
}



document.getElementById("message").addEventListener("keyup", function (event) {
  if (event.keyCode === 13) {
    event.preventDefault();
  }
});

//utils
function get(selector, root = document) {
  return root.querySelector(selector);
}

function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();
  return `${h.slice(-2)}:${m.slice(-2)}`;
}

window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let speech = {
  enabled: true,
  listening: false,
  recognition: new window.SpeechRecognition(),
  text: ''
}

function init() {
  if (('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {

    speech.recognition.continuous = true;
    speech.recognition.interimResults = false;
    speech.recognition.lang = 'id';
    speech.recognition.addEventListener('result', (event) => {
      const audio = event.results[event.results.length - 1];
      speech.text = audio[0].transcript;
      const tag = document.activeElement.nodeName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') {
        if (audio.isFinal) {
          document.activeElement.value += speech.text;
        }
      }
      appendMessage(PERSON_NAME, PERSON_IMG, "right", speech.text);
      botResponse(speech.text)
      console.log(speech.text)
    });

    voicetekan.addEventListener('click', () => {
      speech.listening = !speech.listening;
      if (speech.listening) {
        voicetekan.classList.add('listening');
        voicetekan.style.backgroundColor='green';
        pulse_2.style.borderColor='green'
        speech.recognition.start();
      }
      else  {
        voicetekan.classList.remove('listening');
        voicetekan.style.backgroundColor='red';
        pulse_2.style.borderColor='red'
        speech.recognition.stop();
      }
    })
  }
}
init();

function voiceon() {
  var x = document.getElementById("voicemode");
  if (x.style.display === "none") {
    voicetekan.style.backgroundColor='red';
    pulse_2.style.borderColor='red'
    voicebutton.style.display='none'
    close2.style.display='block'
    x.style.display = "block";
  } else {
    voicebutton.style.display='block'
    close2.style.display='none'
    x.style.display = "none";
    speech.recognition.stop();
  }
}
voiceon()







