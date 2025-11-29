-- Database Migration for Cloud Training
-- Run this SQL to add cloud_training column if it doesn't exist

-- Add cloud_training column to training_jobs table
ALTER TABLE `training_jobs` 
ADD COLUMN IF NOT EXISTS `cloud_training` TINYINT(1) DEFAULT 0 COMMENT '1 if training in cloud (Google Colab), 0 for local';

-- Create training_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS `training_logs` (
  `log_id` INT(11) NOT NULL AUTO_INCREMENT,
  `training_job_id` INT(11) NOT NULL,
  `log_level` VARCHAR(20) NOT NULL DEFAULT 'INFO',
  `message` TEXT NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`log_id`),
  KEY `idx_job_id` (`training_job_id`),
  KEY `idx_created_at` (`created_at`),
  CONSTRAINT `fk_training_logs_job` FOREIGN KEY (`training_job_id`) 
    REFERENCES `training_jobs` (`job_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add index for faster queries
CREATE INDEX IF NOT EXISTS `idx_cloud_training` ON `training_jobs` (`cloud_training`);
CREATE INDEX IF NOT EXISTS `idx_status_cloud` ON `training_jobs` (`status`, `cloud_training`);

