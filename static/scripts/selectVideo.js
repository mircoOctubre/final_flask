const videoPath = document.getElementById("record-route");

const loadItems = ( itemName ) =>{
    item = localStorage.getItem(itemName)
    if(!item){
        return []; 
    }
    return JSON.parse(item);
}

const saveConnectedCamerasLists = ( state ) =>{
    localStorage.setItem("connected_cameras", JSON.stringify(state));
}


const loadVideo = () =>{
    const connectedCamerasList = loadItems("connected_cameras") ;

    connectedCamerasList.push({ 
        id: videoPath.value ,
        name: videoPath.value, 
        model: "video",
        activated: true,
        activeModel: "No_model" ,
        relevantItems: [],
        inferencePercentage : 0
    })

    saveConnectedCamerasLists(connectedCamerasList);

    sendLoadVideoRequest();

    setTimeout(()=>{
        location.replace("/surveillance");
    }, 1000)
}


const sendLoadVideoRequest = () =>{
    const conected_cameras_object = {
        cameras: loadItems("connected_cameras"),
      };

    fetch('/surveillance/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(conected_cameras_object)
      })
}


















