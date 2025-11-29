<?php
/**
 * Cloud Training Helper Functions
 * Integrates Google Colab training with admin interface
 */

// Note: db.php is included by admin_training_module.php
// This file should be included AFTER db.php

/**
 * Start cloud training in Google Colab
 * Called from admin_training_module.php
 */
function startCloudTraining() {
    global $conn;
    
    try {
        // Check if training is already running
        $check_query = "SELECT * FROM training_jobs WHERE status = 'running' LIMIT 1";
        $result = $conn->query($check_query);
        if ($result && $result->num_rows > 0) {
            return ['success' => false, 'message' => 'Training is already running. Please wait for it to complete.'];
        }
        
        // Get training parameters
        $epochs = isset($_POST['epochs']) ? intval($_POST['epochs']) : 50;
        $batch_size = isset($_POST['batch_size']) ? intval($_POST['batch_size']) : 8;
        $training_type = 'object_detection';
        
        // Get admin ID from session
        $admin_id = isset($_SESSION['admin_id']) ? $_SESSION['admin_id'] : 1; // Fallback to 1 if session not available
        
        // Create training job
        // Note: cloud_training column may not exist yet - use IF EXISTS or run migration
        $job_query = "INSERT INTO training_jobs (admin_id, status, started_at, training_type, training_config, cloud_training) 
                     VALUES (?, 'pending', NOW(), ?, ?, 1)";
        $stmt = $conn->prepare($job_query);
        $config = json_encode(['epochs' => $epochs, 'batch_size' => $batch_size]);
        $stmt->bind_param("iss", $admin_id, $training_type, $config);
        $stmt->execute();
        $job_id = $conn->insert_id;
        $stmt->close();
        
        // OPTION 1: Independent Training Service (Heroku/Railway/Render) - RECOMMENDED
        // This runs training completely independently - no browser needed!
        // Set TRAINING_SERVICE_URL environment variable or update here
        $training_service_url = getenv('TRAINING_SERVICE_URL') ?: "https://your-training-service.herokuapp.com"; // UPDATE THIS with your Heroku app URL
        
        if ($training_service_url && $training_service_url !== "https://your-service.railway.app") {
            // Call independent training service
            $ch = curl_init($training_service_url . '/train');
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(['job_id' => $job_id]));
            curl_setopt($ch, CURLOPT_TIMEOUT, 10);
            
            $response = curl_exec($ch);
            $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);
            
            if ($http_code === 200) {
                $result = json_decode($response, true);
                if ($result['success']) {
                    return [
                        'success' => true,
                        'message' => 'Cloud training started automatically! Training runs independently - no browser needed.',
                        'job_id' => $job_id,
                        'service_url' => $training_service_url
                    ];
                }
            }
        }
        
        // OPTION 2: Google Colab (fallback if service not configured)
        // Google Colab notebook URL - UPDATE THIS with your actual notebook ID
        $colab_notebook_id = "YOUR_NOTEBOOK_ID_HERE"; // ⚠️ REPLACE THIS with your actual notebook ID
        $colab_notebook_url = "https://colab.research.google.com/drive/" . $colab_notebook_id . "?usp=sharing";
        
        return [
            'success' => true,
            'message' => 'Cloud training job created! For fully automatic training (no browser), deploy Railway service (see railway_training_service/)',
            'colab_url' => $colab_notebook_url,
            'job_id' => $job_id,
            'instructions' => '1. The notebook will read job ID ' . $job_id . ' from database\n2. Run all cells to start training\n3. Training will use ' . $epochs . ' epochs and batch size ' . $batch_size . '\n\n💡 For fully automatic training (no browser), deploy Railway service (see SETUP_INDEPENDENT_TRAINING.md)'
        ];
        
    } catch (Exception $e) {
        return ['success' => false, 'message' => 'Error: ' . $e->getMessage()];
    }
}

/**
 * Check cloud training status
 * Called from admin_training_module.php
 */
function checkCloudTrainingStatus() {
    global $conn;
    
    try {
        $job_id = isset($_POST['job_id']) ? intval($_POST['job_id']) : (isset($_GET['job_id']) ? intval($_GET['job_id']) : null);
        
        if (!$job_id) {
            return ['success' => false, 'message' => 'Job ID required'];
        }
        
        $stmt = $conn->prepare("SELECT status, completed_at, error_message FROM training_jobs WHERE job_id = ?");
        $stmt->bind_param("i", $job_id);
        $stmt->execute();
        $result = $stmt->get_result();
        $job = $result->fetch_assoc();
        $stmt->close();
        
        return [
            'success' => true,
            'status' => $job['status'] ?? 'unknown',
            'completed_at' => $job['completed_at'] ?? null,
            'error_message' => $job['error_message'] ?? null
        ];
        
    } catch (Exception $e) {
        return ['success' => false, 'message' => 'Error: ' . $e->getMessage()];
    }
}

/**
 * Download model from Google Drive
 */
function downloadModelFromDrive($job_id, $drive_file_id) {
    // This requires Google Drive API setup
    // For now, return instructions
    
    return [
        'success' => true,
        'message' => 'Model ready for download',
        'instructions' => 'Download model from Google Drive and upload via admin interface',
        'drive_file_id' => $drive_file_id
    ];
}

/**
 * Get latest training logs from database
 * Called from admin_training_module.php
 */
function getCloudTrainingLogs() {
    global $conn;
    
    try {
        $job_id = isset($_POST['job_id']) ? intval($_POST['job_id']) : (isset($_GET['job_id']) ? intval($_GET['job_id']) : null);
        $limit = isset($_POST['limit']) ? intval($_POST['limit']) : 50;
        
        if (!$job_id) {
            return ['success' => false, 'message' => 'Job ID required'];
        }
        
        // Check if training_logs table exists, if not return empty
        $table_check = $conn->query("SHOW TABLES LIKE 'training_logs'");
        if (!$table_check || $table_check->num_rows == 0) {
            return ['success' => true, 'logs' => []];
        }
        
        $stmt = $conn->prepare("SELECT log_level, message, created_at FROM training_logs WHERE training_job_id = ? ORDER BY created_at DESC LIMIT ?");
        $stmt->bind_param("ii", $job_id, $limit);
        $stmt->execute();
        $result = $stmt->get_result();
        
        $logs = [];
        while ($row = $result->fetch_assoc()) {
            $logs[] = $row;
        }
        $stmt->close();
        
        return [
            'success' => true,
            'logs' => array_reverse($logs) // Reverse to show oldest first
        ];
        
    } catch (Exception $e) {
        return ['success' => false, 'message' => 'Error: ' . $e->getMessage()];
    }
}

?>

