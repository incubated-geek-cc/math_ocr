document.addEventListener('DOMContentLoaded', async() => {
    console.log('DOMContentLoaded');

    const main = document.querySelector('#main');

    const reset = document.querySelector('#reset');
    const upload = document.querySelector('#upload');
    const dropFileZone = document.querySelector('#dropFileZone');
    const dropFileInnerZone = dropFileZone.querySelector('.card-body');
    const dropFileZoneCaption=document.querySelector('#dropFileZoneCaption');

    const transcriptions=document.querySelectorAll('.transcription');
    var c = document.querySelector('canvas');
	var o = c.getContext('2d');

	function reset_canvas(){
		o.clearRect(0, 0, c.width, c.height);
	}

	function triggerEvent(el, type) {
	    let e = ( ('createEvent' in document) ? document.createEvent('HTMLEvents') : document.createEventObject() );
	    if ('createEvent' in document) { 
	      e.initEvent(type, false, true);
	      el.dispatchEvent(e);
	    } else { 
	      e.eventType = type;
	      el.fireEvent('on' + e.eventType, e);
	    }
	}

	var drag = false, lastX, lastY;
	c.addEventListener((!('ontouchstart' in window)) ? 'mousedown' : 'touchstart', function(e) {
	  	drag = true; 
		lastX = 0; 
		lastY = 0; 
		e.preventDefault(); 
		e.initEvent((!('ontouchstart' in window)) ? 'mousemove' : 'touchmove', false, true);
	});
	
	c.addEventListener((!('ontouchstart' in window)) ? 'mouseup' : 'touchend', function(e){ 
		drag = false; 
		e.preventDefault(); 
	});

	c.addEventListener((!('ontouchstart' in window)) ? 'mousemove' : 'touchmove', function(e){
		e.preventDefault();
		let rect = c.getBoundingClientRect();
		let r = 5;

		function dot(x, y){
			o.beginPath();
			o.moveTo(x + r, y);
			o.arc(x, y, r, 0, Math.PI * 2);
			o.fill();
		}
		if(drag) {
			let x = e.clientX - rect.left;
			let y = e.clientY - rect.top;
			
			if(lastX && lastY){
				let dx = x - lastX, dy = y - lastY;
				let d = Math.sqrt(dx * dx + dy * dy);
				for(let i = 1; i < d; i += 2){
					dot(lastX + dx / d * i, lastY + dy / d * i)
				}
			}
			dot(x, y)
			lastX = x;
			lastY = y;
		}
	});
	reset_canvas();		
 
    function resizeDisplay() {
    	dropFileZoneCaption['style']['margin']=`calc(${dropFileZoneCaption.parentElement.clientHeight/2}px - 0.5rem - ${dropFileZoneCaption.clientHeight/2}px)  auto`;
    	c['style']['margin']=`auto`;
    	
    	c['style']['height']=`calc(${dropFileInnerZone.clientHeight*0.8}px)`; //  - 0.5rem - 0.83em - 0.5rem - 0.83em - 1.5em
    	c['style']['width']=`calc(${dropFileInnerZone.clientWidth*0.8}px)`;

    	c.height=dropFileInnerZone.clientHeight*0.8;
    	c.width=dropFileInnerZone.clientWidth*0.8;
    }
	resizeDisplay();

	window.addEventListener('resize', ()=> {
		resizeDisplay();
	});

	reset.addEventListener('click', ()=> {
		for(let transcription of transcriptions) {
			transcription.innerText='';
			transcription.classList.remove('done');
		}
		dropFileZoneCaption['style']['display']='inline-block';
		dropFileInnerZone['style']['background-image']='';
		dropFileInnerZone['style']['background-repeat']='';
		dropFileInnerZone['style']['background-position']='';
		dropFileInnerZone['style']['background-size']='';

		reset_canvas();
	});

    function readFileAsDataURL(file) {
	    return new Promise((resolve,reject) => {
	        let fileredr = new FileReader();
	        fileredr.onload = () => resolve(fileredr.result);
	        fileredr.onerror = () => reject(fileredr);
	        fileredr.readAsDataURL(file);
	    });
	}

    upload.addEventListener('click', (ev) => {
        ev.currentTarget.value = '';
    });
    dropFileZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropFileInnerZone.classList.add('bg-custom-two-05');
    });
    dropFileZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropFileInnerZone.classList.remove('bg-custom-two-05');
    });
    dropFileZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropFileInnerZone.classList.add('bg-custom-two-05');
    });

    dropFileZone.addEventListener("drop", async(e) => {
        e.preventDefault();
        e.stopPropagation();
        dropFileInnerZone.classList.remove("bg-custom-two-05");
        upload.value = '';

        let draggedData = e.dataTransfer;
        let file = draggedData.files[0];
        if (!file) return;
        await importFile(file);
    }); // drop file change event

    upload.addEventListener('change', async(evt) => {
        const file=evt.currentTarget.files[0];
		if (!file) return;
        await importFile(file);
    }); // upload file change event

    function selectCopyText(node) { // inline-math
        let isVal=true;
        node = document.querySelector(`#${node}`);
        try {
            node.select();
            try {
                node.setSelectionRange(0, 99999); /* For mobile devices */
            } catch(err0) {}
        } catch(err) {
            isVal=false;
            console.log(err.message);
            if (document.body.createTextRange) {
                const range = document.body.createTextRange();
                range.moveToElementText(node);
                range.select();
            } else if (window.getSelection) {
                const selection = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(node);
                selection.removeAllRanges();
                selection.addRange(range);
            } else {
                console.warn('Could not select text in node: Unsupported browser.');
            }
        } finally {
            navigator.clipboard.writeText(isVal ? node.value : node.innerText);
        }
    }

    const copyBtns = document.querySelectorAll('.copy-btn');
    for(let copyBtn of copyBtns) {
    	copyBtn.addEventListener('click', (evt)=> {
    		let eleID=evt.currentTarget.value;
    		selectCopyText(eleID);
    	});
    }
	const loadImage = (url) => new Promise((resolve, reject) => {
      const img = new Image();
      img.addEventListener('load', () => resolve(img));
      img.addEventListener('error', (err) => reject(err));
      img.src = url;
    });

   	async function importFile(file) {
    	try {
	    	let fileName=file.name;
			let fileType=file.type;
			let fileSizeInKB=parseInt(file.size/1024);
			let fileSizeInMB=((file.size/1024)/1024).toFixed(2);
			
	        let b64Str = await readFileAsDataURL(file);
			let _img = await loadImage(b64Str);

			let imgW=_img.naturalWidth;
			let imgH=_img.naturalHeight;

			dropFileZoneCaption['style']['display']='none';
			dropFileInnerZone['style']['background-image']=`url("${b64Str}")`;
			dropFileInnerZone['style']['background-repeat']='no-repeat';
			dropFileInnerZone['style']['background-position']='center';
			dropFileInnerZone['style']['background-size']='contain';

			await recognize_image(b64Str);
        } catch (err) {
            alert(`⚠ ERROR: ${err.message}`);
            console.log(err);
        }
        return await Promise.resolve('success');
    }
  	
  	const ocrDrawing = document.querySelector('#ocrDrawing');
  	ocrDrawing.addEventListener('click', async(evt)=> {
  		let b64Str=c.toDataURL();
  		// console.log(b64Str);
  		await recognize_image(b64Str);
  	});

  	const toggle_view = document.querySelector('form#toggle_view');
  	toggle_view.addEventListener('change', (evt)=> {
  		let ele = evt.srcElement;
  		let viewContainers=document.querySelectorAll('div[data-view]');
  		viewContainers[0].setAttribute('hidden','');
  		viewContainers[1].setAttribute('hidden','');
 
  		let viewContainer=document.querySelector(`div[data-view="${ele.id}"]`);
  		viewContainer.removeAttribute('hidden');
  	});

  	const latexHTML=document.querySelector('#latex-html');

    async function recognize_image(b64Str) {
    	b64Str = b64Str.replace('data:image/png;base64,','');
    	let onnxTranscription = document.querySelector('.transcription[data-ocr="onnx"]');
    	onnxTranscription.innerText = '(Recognizing...)';

    	let latexTranscription = document.querySelector('.transcription[data-ocr="latex"]');
    	latexTranscription.innerText = '(Recognizing...)';

    	const response = await fetch(`./math_ocr/post`, {
    		method: 'POST',
    		body: JSON.stringify({
    			'im_b64': b64Str
    		}),
    		headers: {
    			'Content-type': 'application/json; charset=UTF-8'
    		}
    	});
	  	const result = await response.json();
	  	let inlineTxt = result['result'];
		onnxTranscription.classList.add('done');
		onnxTranscription.innerText = inlineTxt;

		katex.render(inlineTxt, latexTranscription, {
		    throwOnError: false
		});
		latexTranscription.classList.add('done');

		let html = katex.renderToString(inlineTxt, {
		    throwOnError: false
		});
		latexHTML.value=html;
	}
	
});