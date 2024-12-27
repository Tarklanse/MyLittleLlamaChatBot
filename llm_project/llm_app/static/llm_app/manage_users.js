// Function to handle Edit User
async function saveUserChanges() {
    const oldUsername = document.getElementById('editUsername').getAttribute('data-old-username');
    const newUsername = document.getElementById('editUsername').value;
    const newPassword = document.getElementById('editPassword').value;
    const role = document.getElementById('editRole').value;

    // Send API request to edit user
    const response = await fetch('/manage/users', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
        },
        body: JSON.stringify({
            action: 'edit',
            old_username: oldUsername,
            new_username: newUsername,
            new_password: newPassword,
            role: role,
            password: ''  // Add if password change is needed
        })
    });

    const result = await response.json();

    if (result.status === 'success') {
        alert('User updated successfully!');
        window.location.reload(); // Refresh the page to show updated user list
    } else {
        alert('Error updating user: ' + result.message);
    }

    closeEditPopup();
}

// Function to handle Delete User
async function deleteUser(username) {
    if (!confirm(`Are you sure you want to delete the user: ${username}?`)) {
        return;
    }

    // Send API request to delete user
    const response = await fetch('/manage/users', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
        },
        body: JSON.stringify({
            action: 'delete',
            username: username
        })
    });

    const result = await response.json();

    if (result.status === 'success') {
        alert('User deleted successfully!');
        window.location.reload(); // Refresh the page to show updated user list
    } else {
        alert('Error deleting user: ' + result.message);
    }
}

// Utility function to open the Edit User popup
function openEditPopup(username, role) {
    const popup = document.getElementById('editPopup');
    const usernameField = document.getElementById('editUsername');

    usernameField.value = username;
    usernameField.setAttribute('data-old-username', username); // Save the original username
    document.getElementById('editRole').value = role;

    popup.classList.remove('hidden');
}

// Utility function to close the Edit User popup
function closeEditPopup() {
    const popup = document.getElementById('editPopup');
    popup.classList.add('hidden');
}

// Utility function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.startsWith(name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Function to open the Add User popup
function openAddPopup() {
    const popup = document.getElementById('addPopup');
    document.getElementById('addUsername').value = '';
    document.getElementById('addRole').value = '';
    document.getElementById('addPassword').value = ''; // Clear input fields
    popup.classList.remove('hidden');
}

// Function to close the Add User popup
function closeAddPopup() {
    const popup = document.getElementById('addPopup');
    popup.classList.add('hidden');
}

// Function to handle adding a new user
async function addUser() {
    const username = document.getElementById('addUsername').value;
    const role = document.getElementById('addRole').value;
    const password = document.getElementById('addPassword').value;

    if (!username || !password) {
        alert('Username and password are required.');
        return;
    }

    // Send API request to add user
    const response = await fetch('/manage/users', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')  // CSRF token for security
        },
        body: JSON.stringify({
            action: 'add',
            username: username,
            password: password,
            role: role
        })
    });

    const result = await response.json();

    if (result.status === 'success') {
        alert('User added successfully!');
        window.location.reload(); // Refresh the page to show updated user list
    } else {
        alert('Error adding user: ' + result.message);
    }

    closeAddPopup();
}

