async function generateKolam() {
      const seed = document.getElementById("prompt").value.trim();
      if (!seed) {
        alert("Please enter a seed first.");
        return;
      }

      // Clear previous animation and hide placeholder immediately
      stopAnimation();
      clearCanvas();
      document.getElementById('placeholderText').style.display = 'none';
      
      try {
        // Get drawing steps from backend for animation
        const response = await fetch(`/drawkolam_steps?seed=${encodeURIComponent(seed)}&depth=2`);
        if (!response.ok) {
          throw new Error('Failed to generate kolam steps');
        }
        
        const data = await response.json();
        drawingSteps = data.steps;
        currentStep = 0;
        
        console.log('Drawing steps loaded:', drawingSteps.length);
        
        // Update progress and start animation
        updateProgressInfo();
        startAnimation();
        
        // Also get narration if the function exists
        if (typeof getSeedNarration === 'function') {
          await getSeedNarration(seed);
        }
        
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate kolam animation');
        // Show placeholder again if there's an error
        document.getElementById('placeholderText').style.display = 'block';
      }
    }

async function getSeedNarration(seed){
  const panel = document.getElementById('narrationPanel');
  const box = document.getElementById('seedNarration');
  try {
    panel.style.display='block';
    box.classList.add('loading');
    box.textContent='Generating narration...';
    const resp = await fetch(`/narrate_seed?seed=${encodeURIComponent(seed)}&depth=2`);
    const data = await resp.json();
    box.classList.remove('loading');
    if(data.success && data.narration){
      box.textContent=data.narration;
    } else {
      box.textContent='Error: '+(data.error || 'Unable to narrate this seed.');
    }
  } catch(err){
    box.classList.remove('loading');
    box.textContent='Error: Failed to connect to server.';
  }
}

async function generateFromPrompt() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) {
    alert('Please enter a prompt first.');
    return;
  }

  const loader = document.getElementById('loader');
  const promptInput = document.getElementById('prompt');
  const promptButton = document.getElementById('prompt-button');

  loader.style.display = 'block';
  promptButton.disabled = true;

  try {
    const response = await fetch('/generate_seed_from_prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to generate seed.');
    }

    const data = await response.json();
    if(data.seed){
      promptInput.value = data.seed;
      generateKolam();
    } else if(data.error){
      alert('Failed: '+data.error);
    }
  } catch (error) {
    console.error('Error:', error);
    alert(`An error occurred: ${error.message}`);
  } finally {
    loader.style.display = 'none';
    promptButton.disabled = false;
  }
}
