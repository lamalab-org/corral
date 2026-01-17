// Add copy buttons to all code blocks
document.addEventListener('DOMContentLoaded', function() {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('pre > code');

    codeBlocks.forEach(function(codeBlock) {
        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy code to clipboard');

        // Create wrapper for positioning
        const pre = codeBlock.parentElement;
        pre.style.position = 'relative';

        // Add button to pre element
        pre.appendChild(button);

        // Add click handler
        button.addEventListener('click', function() {
            const code = codeBlock.textContent;

            // Use Clipboard API
            navigator.clipboard.writeText(code).then(function() {
                // Success feedback
                button.textContent = 'Copied!';
                button.classList.add('copied');

                // Reset after 2 seconds
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            }).catch(function(err) {
                // Fallback for older browsers
                console.error('Failed to copy:', err);
                button.textContent = 'Failed';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
    });
});
