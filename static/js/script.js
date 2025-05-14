document.addEventListener('DOMContentLoaded', function() {
    // Handle symptom selection
    const symptomSelect = document.getElementById('symptom-select');
    const selectedSymptoms = document.getElementById('selected-symptoms');
    const symptomInput = document.getElementById('symptom-input');
    const symptomCountDisplay = document.getElementById('symptom-count');
    
    // Form submission loaders
    const diseaseForm = document.getElementById('disease-prediction-form');
    const skinForm = document.getElementById('skin-disease-form');
    const diseaseLoader = document.getElementById('disease-loader');
    const skinLoader = document.getElementById('skin-loader');
    
    // Image preview
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    
    // Track selected symptoms
    let selectedSymptomsList = [];
    
    // Initialize any pre-selected symptoms from previous prediction
    const preselectedItems = document.querySelectorAll('.symptom-badge');
    preselectedItems.forEach(item => {
        const symptomName = item.getAttribute('data-symptom');
        if (symptomName) {
            selectedSymptomsList.push(symptomName);
        }
    });
    updateSymptomCount();
    
    // Handle adding symptoms to selection
    if (symptomSelect) {
        symptomSelect.addEventListener('change', function() {
            const symptom = this.value;
            if (symptom && !selectedSymptomsList.includes(symptom)) {
                addSymptom(symptom);
                this.value = ''; // Reset select
            }
        });
    }
    
    // Handle removing symptoms from selection
    if (selectedSymptoms) {
        selectedSymptoms.addEventListener('click', function(e) {
            if (e.target.classList.contains('badge-close')) {
                const symptomBadge = e.target.closest('.symptom-badge');
                if (symptomBadge) {
                    const symptom = symptomBadge.getAttribute('data-symptom');
                    removeSymptom(symptom);
                    symptomBadge.remove();
                }
            }
        });
    }
    
    // Handle form submissions with loading indicators
    if (diseaseForm) {
        diseaseForm.addEventListener('submit', function() {
            if (selectedSymptomsList.length === 0) {
                alert('Please select at least one symptom');
                return false;
            }
            diseaseLoader.style.display = 'block';
        });
    }
    
    if (skinForm) {
        skinForm.addEventListener('submit', function() {
            if (!imageUpload.files[0]) {
                alert('Please select an image to upload');
                return false;
            }
            skinLoader.style.display = 'block';
        });
    }
    
    // Handle image preview
    if (imageUpload) {
        imageUpload.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.innerHTML = `<img src="${e.target.result}" class="uploaded-image" alt="Image preview">`;
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Function to add a symptom to the selection
    function addSymptom(symptom) {
        if (!selectedSymptomsList.includes(symptom)) {
            selectedSymptomsList.push(symptom);
            
            const badge = document.createElement('div');
            badge.className = 'badge bg-primary me-2 mb-2 symptom-badge';
            badge.setAttribute('data-symptom', symptom);
            badge.innerHTML = `${symptom} <span class="badge-close">&times;</span>`;
            
            selectedSymptoms.appendChild(badge);
            
            // Add the symptom to the hidden input
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'symptoms';
            input.value = symptom;
            symptomInput.appendChild(input);
            
            updateSymptomCount();
        }
    }
    
    // Function to remove a symptom from the selection
    function removeSymptom(symptom) {
        const index = selectedSymptomsList.indexOf(symptom);
        if (index !== -1) {
            selectedSymptomsList.splice(index, 1);
            
            // Remove the hidden input
            const inputs = symptomInput.querySelectorAll('input');
            inputs.forEach(input => {
                if (input.value === symptom) {
                    input.remove();
                }
            });
            
            updateSymptomCount();
        }
    }
    
    // Update the symptom count display
    function updateSymptomCount() {
        if (symptomCountDisplay) {
            symptomCountDisplay.textContent = `${selectedSymptomsList.length} symptom(s) selected`;
        }
    }
    
    // Bootstrap tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
