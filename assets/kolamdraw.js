async function generateKolam() {
  const seed = document.getElementById('prompt').value.trim();
  if(!seed){
    alert('Please enter a seed first.');
    return;
  }
  const ts = Date.now();
  const url = `/drawkolam?seed=${encodeURIComponent(seed)}&depth=2&ts=${ts}`;
  const img = document.getElementById('kolamImage');
  img.src = url;
  img.classList.remove('hidden');
  document.getElementById('placeholderText').classList.add('hidden');
  await getSeedNarration(seed);
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
