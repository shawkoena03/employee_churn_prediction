# =============================================================
# PHASE 5 — RETENTION NOTIFICATION SERVICE
# Project : Employee Churn Prediction
# Description : Sends beautifully styled HTML email notifications
#               about at-risk employees to HR.
# =============================================================

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# =============================================================
# ENV LOADER (Custom robust fallback parser - no dotenv required)
# =============================================================
def load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    val = value.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key.strip()] = val

load_env()

# Default configuration from environment variables
SMTP_SERVER   = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "rainbowsweet0308@gmail.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "ruudlwhovdvizurq")
SENDER_EMAIL  = os.environ.get("SENDER_EMAIL", "rainbowsweet0308@gmail.com")
HR_EMAIL      = os.environ.get("HR_RECEIVER_EMAIL", "rainbowsweet0308@gmail.com")

# =============================================================
# HTML EMAIL GENERATOR
# =============================================================
def generate_html_email(employee_id: int,
                        risk_level: str,
                        probability: float,
                        recommendations: list,
                        actual_label: int) -> str:
    """
    Generates a beautifully styled, premium-designed HTML email body with CSS.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Visual aesthetics based on risk levels
    color_map = {
        'CRITICAL': '#D32F2F',  # Dark Red
        'HIGH': '#F57C00',      # Dark Orange
        'MEDIUM': '#FBC02D',    # Dark Yellow
        'LOW': '#388E3C'        # Dark Green
    }
    bg_map = {
        'CRITICAL': '#FFEBEE',
        'HIGH': '#FFF3E0',
        'MEDIUM': '#FFFDE7',
        'LOW': '#E8F5E9'
    }
    
    risk_color = color_map.get(risk_level, '#333333')
    risk_bg = bg_map.get(risk_level, '#F5F5F5')
    actual_str = "LEFT (Attrited) ✗" if actual_label == 1 else "STAYED (Active) ✓"
    actual_color = "#D32F2F" if actual_label == 1 else "#388E3C"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #333333;
                background-color: #F9F9F9;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 24px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                margin: 0 auto;
            }}
            .header {{
                border-bottom: 2px solid #E0E0E0;
                padding-bottom: 12px;
                margin-bottom: 20px;
            }}
            .logo {{
                font-size: 20px;
                font-weight: bold;
                color: #1976D2;
            }}
            .alert-banner {{
                background-color: {risk_bg};
                border-left: 5px solid {risk_color};
                padding: 16px;
                border-radius: 4px;
                margin-bottom: 24px;
            }}
            .risk-badge {{
                display: inline-block;
                background-color: {risk_color};
                color: #FFFFFF;
                padding: 4px 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
                margin-top: 6px;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 24px;
            }}
            .metrics-table td {{
                padding: 10px;
                border-bottom: 1px solid #EEEEEE;
            }}
            .metrics-table .label {{
                font-weight: bold;
                color: #555555;
                width: 40%;
            }}
            .metrics-table .value {{
                text-align: right;
                font-weight: bold;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: bold;
                border-bottom: 1px solid #EEEEEE;
                padding-bottom: 6px;
                margin-top: 24px;
                margin-bottom: 14px;
                color: #1976D2;
            }}
            .recommendation-card {{
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 14px;
                margin-bottom: 12px;
                background-color: #FAFAFA;
            }}
            .priority-urgent {{
                color: #D32F2F;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .priority-rec {{
                color: #1976D2;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .action-text {{
                margin-top: 6px;
                font-size: 13px;
                color: #444444;
                line-height: 1.4;
            }}
            .footer {{
                margin-top: 30px;
                border-top: 1px solid #EEEEEE;
                padding-top: 12px;
                font-size: 11px;
                color: #888888;
                text-align: center;
                line-height: 1.5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🎯 RETENTION INTELLIGENCE AGENT</div>
            </div>
            
            <div class="alert-banner">
                <h2 style="margin: 0; color: {risk_color};">Churn Risk Alert: Employee #{employee_id}</h2>
                <div class="risk-badge">{risk_level} RISK LEVEL</div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <td class="label">Analysis Date</td>
                    <td class="value">{now}</td>
                </tr>
                <tr>
                    <td class="label">Churn Probability</td>
                    <td class="value" style="color: {risk_color}; font-size: 18px;">{probability:.1%}</td>
                </tr>
                <tr>
                    <td class="label">Historical Ground Truth</td>
                    <td class="value" style="color: {actual_color};">{actual_str}</td>
                </tr>
            </table>
            
            <div class="section-title">TOP DRIVERS & TAILORED ACTIONS</div>
    """
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            p_class = "priority-urgent" if rec['priority'] == 'URGENT' else "priority-rec"
            html += f"""
            <div class="recommendation-card">
                <span class="{p_class}">[{rec['priority']}] {i}. {rec['feature'].replace('_', ' ')}</span>
                <div style="font-size: 12px; color: #666666; margin-top: 4px;">
                    Current Value: <strong>{rec['value_label']}</strong> | SHAP Impact: <strong>{rec['shap_score']:+.4f}</strong>
                </div>
                <div class="action-text"><strong>HR Recommended Action:</strong> {rec['action']}</div>
            </div>
            """
    else:
        html += "<p style='font-size: 13px; color: #666666;'>No specific feature-driven risk factors identified.</p>"
        
    html += f"""
            <div class="section-title">GENERAL MANAGER RETENTION PLAN</div>
            <ul style="font-size: 13px; color: #444444; padding-left: 20px; line-height: 1.5;">
    """
    
    if risk_level == 'CRITICAL':
        html += """
                <li>⚡ <strong>Immediate Action:</strong> Schedule a face-to-face retention meeting within 24 hours.</li>
                <li>⚡ <strong>Escalation:</strong> Notify the direct department head and senior HR leadership.</li>
        """
    elif risk_level == 'HIGH':
        html += """
                <li>⚠️ <strong>Urgent Action:</strong> Schedule a 1-on-1 career pathing check-in within 3 days.</li>
                <li>⚠️ <strong>Review:</strong> Conduct compensation analysis against market-rate benchmark data.</li>
        """
    elif risk_level == 'MEDIUM':
        html += """
                <li>📋 <strong>Monitoring:</strong> Add to the monthly HR retention review dashboard.</li>
                <li>📋 <strong>Engagement:</strong> Sync with the employee's direct manager about team engagement indicators.</li>
        """
    else:
        html += """
                <li>✅ <strong>Maintenance:</strong> Maintain regular touchpoints and quarterly career reviews.</li>
        """
        
    html += """
            </ul>
            
            <div class="footer">
                This is an automated notification from the Employee Churn Prediction Platform.<br/>
                Please do not reply directly to this mail transmission.<br/>
                &copy; 2026 Retention Intelligence Agent Inc.
            </div>
        </div>
    </body>
    </html>
    """
    return html

# =============================================================
# EMAIL SENDER CLIENT
# =============================================================
def send_retention_email(employee_id: int,
                         risk_level: str,
                         probability: float,
                         recommendations: list,
                         actual_label: int,
                         custom_sender: str = None,
                         custom_receiver: str = None) -> tuple:
    """
    Builds the MIMEMultipart container and sends the real-time email via SMTP.
    If credentials are missing or the connection fails, it falls back to
    simulating and saving the email as an HTML file locally for review.
    
    Returns (success_bool, message_string)
    """
    sender = custom_sender if custom_sender else SENDER_EMAIL
    receiver = custom_receiver if custom_receiver else HR_EMAIL
    
    # Compile the HTML text body
    html_body = generate_html_email(
        employee_id=employee_id,
        risk_level=risk_level,
        probability=probability,
        recommendations=recommendations,
        actual_label=actual_label
    )
    
    # 1. Fallback to Local Simulation Mode if credentials are not configured
    is_placeholder = (SMTP_USERNAME == "" or "your_username" in SMTP_USERNAME or SMTP_PASSWORD == "")
    if is_placeholder:
        # Create output directory
        report_dir = os.path.join("outputs", "simulated_emails")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"simulated_email_employee_{employee_id}.html")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_body)
            
        success_msg = (
            f"No SMTP credentials found in '.env' (Placeholder detected).\n"
            f"SUCCESSFULLY simulated email locally!\n"
            f"Saved HTML to: outputs/simulated_emails/simulated_email_employee_{employee_id}.html"
        )
        print(success_msg)
        return True, f"Simulated locally (saved file: simulated_email_employee_{employee_id}.html)"
        
    # 2. Real SMTP delivery mode
    try:
        # Set up message container
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"{'🔴' if risk_level == 'CRITICAL' else '🟠'} [{risk_level} RISK] Employee #{employee_id} — Churn Alert"
        msg['From']    = sender
        msg['To']      = receiver
        
        # Attach HTML content
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to mail server
        print(f"Connecting to SMTP mail server: {SMTP_SERVER}:{SMTP_PORT}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if SMTP_PORT == 587:
                server.starttls()  # Upgrade secure connection
                server.ehlo()
            
            print(f"Authenticating as: {SMTP_USERNAME}...")
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            
            print(f"Delivering email to recipient: {receiver}...")
            server.sendmail(sender, receiver, msg.as_string())
            
        print("✅ Email delivered successfully!")
        return True, "Email sent successfully to SMTP server"
        
    except Exception as e:
        error_msg = f"SMTP transmission failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Secondary fallback: save locally if connection error occurred
        report_dir = os.path.join("outputs", "simulated_emails")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"failed_smtp_employee_{employee_id}.html")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_body)
            
        return False, f"Connection failed. Local backup saved: outputs/simulated_emails/failed_smtp_employee_{employee_id}.html (Error: {str(e)})"
