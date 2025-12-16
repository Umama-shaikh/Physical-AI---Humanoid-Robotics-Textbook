// Inject JavaScript to make navbar title clickable
export default function (context, options) {
  return {
    name: 'navbar-title-click-plugin',
    injectHtmlTags() {
      return {
        postBodyTags: [
          `<script>
            document.addEventListener('DOMContentLoaded', function() {
              // Find the navbar title element and make it clickable
              const titleElement = document.querySelector('.navbar__title');
              if (titleElement) {
                titleElement.style.cursor = 'pointer';
                titleElement.setAttribute('tabindex', '0'); // Make keyboard accessible
                titleElement.setAttribute('role', 'button'); // ARIA role
                titleElement.setAttribute('aria-label', 'Go to homepage'); // ARIA label
                titleElement.addEventListener('click', function() {
                  window.location.href = '/';
                });
                // Add keyboard support (Enter and Space keys)
                titleElement.addEventListener('keydown', function(e) {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    window.location.href = '/';
                  }
                });
              }

              // Find the navbar logo element and make it clickable
              const logoElement = document.querySelector('.navbar__logo img');
              if (logoElement) {
                logoElement.style.cursor = 'pointer';
                logoElement.setAttribute('tabindex', '0'); // Make keyboard accessible
                logoElement.setAttribute('role', 'button'); // ARIA role
                logoElement.setAttribute('aria-label', 'Go to homepage'); // ARIA label
                // Add click handler to parent anchor if it exists, otherwise to the image
                const parentLink = logoElement.closest('a');
                if (parentLink) {
                  parentLink.style.cursor = 'pointer';
                  parentLink.addEventListener('click', function(e) {
                    // Only navigate to root if not already navigating to another link
                    if (window.location.pathname !== '/') {
                      e.preventDefault();
                      window.location.href = '/';
                    }
                  });
                  // Add keyboard support (Enter and Space keys)
                  parentLink.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      if (window.location.pathname !== '/') {
                        window.location.href = '/';
                      }
                    }
                  });
                } else {
                  logoElement.addEventListener('click', function() {
                    window.location.href = '/';
                  });
                  // Add keyboard support (Enter and Space keys)
                  logoElement.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      window.location.href = '/';
                    }
                  });
                }
              }
            });
          </script>`
        ],
      };
    },
  };
}