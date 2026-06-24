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

async function deleteVector(chatId, vectorId) {
    if (!confirm(`Delete vector for chat: ${chatId}?\nThis will permanently remove the vector collection.`)) {
        return;
    }

    const response = await fetch('/manage/vectors', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            action: 'delete',
            chat_id: chatId,
            vector_id: vectorId
        })
    });

    const result = await response.json();

    if (result.status === 'success') {
        alert('Vector deleted successfully!');
        window.location.reload();
    } else {
        alert('Error deleting vector: ' + (result.message || 'Unknown error'));
    }
}

async function resetAllVectors() {
    if (!confirm('Are you sure you want to reset ALL vectors?\nThis action will permanently delete all stored vector data and cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/manage/vectors_full_reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            }
        });

        const result = await response.json();
        if (result.status === 'success') {
            alert('All vectors have been reset successfully!');
            window.location.reload();
        } else {
            alert('Error resetting vectors: ' + (result.message || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error during vector reset:', error);
        alert('An error occurred while trying to reset vectors.');
    }
}
