"""
Security module for Drift-ML platform.
Provides functionality for authentication, authorization, encryption, and audit logging.
"""
import os
import json
import time
import logging
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import jwt
import requests
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Base class for database models
Base = declarative_base()

# User model
class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(64), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

# Role model
class Role(Base):
    """Role model for role-based access control."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# User-Role association model
class UserRole(Base):
    """User-Role association model."""
    
    __tablename__ = "user_roles"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    role_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Permission model
class Permission(Base):
    """Permission model for role-based access control."""
    
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255), nullable=True)
    resource = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Role-Permission association model
class RolePermission(Base):
    """Role-Permission association model."""
    
    __tablename__ = "role_permissions"
    
    id = Column(Integer, primary_key=True)
    role_id = Column(Integer, nullable=False)
    permission_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Audit log model
class AuditLog(Base):
    """Audit log model for tracking actions."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    username = Column(String(100), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=False)
    resource_id = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Authentication:
    """Authentication module using JWT and OAuth2/OIDC."""
    
    def __init__(
        self,
        db_url: str,
        jwt_secret: str,
        token_expiry: int = 3600,
        oauth_providers: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the authentication module.
        
        Args:
            db_url: Database URL for user storage
            jwt_secret: Secret for JWT token signing
            token_expiry: Token expiry time in seconds
            oauth_providers: Dictionary of OAuth providers and their configurations
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.jwt_secret = jwt_secret
        self.token_expiry = token_expiry
        self.oauth_providers = oauth_providers or {}
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        is_admin: bool = False
    ) -> Optional[int]:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            is_admin: Whether the user is an administrator
            
        Returns:
            User ID or None if registration failed
        """
        session = self.Session()
        
        try:
            # Check if username or email already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                logger.warning(f"User with username '{username}' or email '{email}' already exists")
                return None
            
            # Generate salt and hash password
            salt = secrets.token_hex(32)
            password_hash = self._hash_password(password, salt)
            
            # Create new user
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                is_admin=is_admin
            )
            
            session.add(new_user)
            session.commit()
            
            logger.info(f"Registered new user '{username}'")
            return new_user.id
        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User information dictionary or None if authentication failed
        """
        session = self.Session()
        
        try:
            # Get user by username
            user = session.query(User).filter(User.username == username).first()
            
            if not user:
                logger.warning(f"User '{username}' not found")
                return None
            
            # Check password
            if not self._verify_password(password, user.salt, user.password_hash):
                logger.warning(f"Invalid password for user '{username}'")
                return None
            
            # Update last login timestamp
            user.last_login = datetime.utcnow()
            session.commit()
            
            # Return user information
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def generate_token(
        self,
        user_id: int,
        username: str,
        is_admin: bool = False,
        custom_claims: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate a JWT token for the user.
        
        Args:
            user_id: User ID
            username: Username
            is_admin: Whether the user is an administrator
            custom_claims: Custom claims to include in the token
            
        Returns:
            JWT token or None if token generation failed
        """
        try:
            # Create payload
            now = datetime.utcnow()
            payload = {
                "sub": str(user_id),
                "username": username,
                "is_admin": is_admin,
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(seconds=self.token_expiry)).timestamp())
            }
            
            # Add custom claims
            if custom_claims:
                payload.update(custom_claims)
            
            # Generate token
            token = jwt.encode(
                payload,
                self.jwt_secret,
                algorithm="HS256"
            )
            
            return token
        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload or None if verification failed
        """
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None
    
    def oauth_login(
        self,
        provider: str,
        code: str,
        redirect_uri: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with OAuth.
        
        Args:
            provider: OAuth provider (e.g., "google", "github")
            code: Authorization code
            redirect_uri: Redirect URI
            
        Returns:
            User information and token or None if authentication failed
        """
        try:
            if provider not in self.oauth_providers:
                logger.warning(f"Unsupported OAuth provider: {provider}")
                return None
            
            # Get provider config
            provider_config = self.oauth_providers[provider]
            
            # Exchange code for token
            token_url = provider_config["token_url"]
            client_id = provider_config["client_id"]
            client_secret = provider_config["client_secret"]
            
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            
            tokens = token_response.json()
            access_token = tokens.get("access_token")
            
            if not access_token:
                logger.warning("No access token in OAuth response")
                return None
            
            # Get user information
            userinfo_url = provider_config["userinfo_url"]
            headers = {"Authorization": f"Bearer {access_token}"}
            
            userinfo_response = requests.get(userinfo_url, headers=headers)
            userinfo_response.raise_for_status()
            
            userinfo = userinfo_response.json()
            
            # Extract user details based on provider
            if provider == "google":
                email = userinfo.get("email")
                username = email.split("@")[0] if email else None
                name = userinfo.get("name")
            elif provider == "github":
                username = userinfo.get("login")
                email = userinfo.get("email")
                name = userinfo.get("name")
            else:
                username = userinfo.get("preferred_username") or userinfo.get("sub")
                email = userinfo.get("email")
                name = userinfo.get("name")
            
            if not username or not email:
                logger.warning("Incomplete user information from OAuth provider")
                return None
            
            # Check if user exists
            session = self.Session()
            
            try:
                user = session.query(User).filter(User.email == email).first()
                
                if not user:
                    # Create new user
                    salt = secrets.token_hex(32)
                    password_hash = self._hash_password(secrets.token_urlsafe(32), salt)
                    
                    user = User(
                        username=username,
                        email=email,
                        password_hash=password_hash,
                        salt=salt,
                        is_active=True
                    )
                    
                    session.add(user)
                    session.commit()
                
                # Update last login
                user.last_login = datetime.utcnow()
                session.commit()
                
                # Generate token
                token = self.generate_token(
                    user_id=user.id,
                    username=user.username,
                    is_admin=user.is_admin,
                    custom_claims={"oauth_provider": provider}
                )
                
                return {
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "is_admin": user.is_admin
                    },
                    "token": token
                }
            except Exception as e:
                logger.error(f"Failed to process OAuth login: {e}")
                session.rollback()
                return None
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to authenticate with OAuth: {e}")
            return None
    
    def _hash_password(self, password: str, salt: str) -> str:
        """
        Hash a password with a salt.
        
        Args:
            password: Password to hash
            salt: Salt for hashing
            
        Returns:
            Password hash
        """
        password_salt = (password + salt).encode()
        return hashlib.sha256(password_salt).hexdigest()
    
    def _verify_password(self, password: str, salt: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Password to verify
            salt: Salt used for hashing
            password_hash: Stored password hash
            
        Returns:
            True if password is correct, False otherwise
        """
        return self._hash_password(password, salt) == password_hash

class Authorization:
    """Role-based access control module."""
    
    def __init__(self, db_url: str):
        """
        Initialize the authorization module.
        
        Args:
            db_url: Database URL for permission storage
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_role(
        self,
        name: str,
        description: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            
        Returns:
            Role ID or None if creation failed
        """
        session = self.Session()
        
        try:
            # Check if role already exists
            existing_role = session.query(Role).filter(Role.name == name).first()
            
            if existing_role:
                logger.warning(f"Role '{name}' already exists")
                return existing_role.id
            
            # Create new role
            new_role = Role(
                name=name,
                description=description
            )
            
            session.add(new_role)
            session.commit()
            
            logger.info(f"Created role '{name}'")
            return new_role.id
        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def create_permission(
        self,
        name: str,
        resource: str,
        action: str,
        description: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new permission.
        
        Args:
            name: Permission name
            resource: Resource the permission applies to
            action: Action the permission allows
            description: Permission description
            
        Returns:
            Permission ID or None if creation failed
        """
        session = self.Session()
        
        try:
            # Check if permission already exists
            existing_permission = session.query(Permission).filter(
                Permission.name == name
            ).first()
            
            if existing_permission:
                logger.warning(f"Permission '{name}' already exists")
                return existing_permission.id
            
            # Create new permission
            new_permission = Permission(
                name=name,
                resource=resource,
                action=action,
                description=description
            )
            
            session.add(new_permission)
            session.commit()
            
            logger.info(f"Created permission '{name}'")
            return new_permission.id
        except Exception as e:
            logger.error(f"Failed to create permission: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def assign_role_to_user(
        self,
        user_id: int,
        role_id: int
    ) -> bool:
        """
        Assign a role to a user.
        
        Args:
            user_id: User ID
            role_id: Role ID
            
        Returns:
            True if assignment succeeded, False otherwise
        """
        session = self.Session()
        
        try:
            # Check if assignment already exists
            existing_assignment = session.query(UserRole).filter(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id
            ).first()
            
            if existing_assignment:
                logger.warning(f"User {user_id} already has role {role_id}")
                return True
            
            # Create new assignment
            new_assignment = UserRole(
                user_id=user_id,
                role_id=role_id
            )
            
            session.add(new_assignment)
            session.commit()
            
            logger.info(f"Assigned role {role_id} to user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def assign_permission_to_role(
        self,
        role_id: int,
        permission_id: int
    ) -> bool:
        """
        Assign a permission to a role.
        
        Args:
            role_id: Role ID
            permission_id: Permission ID
            
        Returns:
            True if assignment succeeded, False otherwise
        """
        session = self.Session()
        
        try:
            # Check if assignment already exists
            existing_assignment = session.query(RolePermission).filter(
                RolePermission.role_id == role_id,
                RolePermission.permission_id == permission_id
            ).first()
            
            if existing_assignment:
                logger.warning(f"Role {role_id} already has permission {permission_id}")
                return True
            
            # Create new assignment
            new_assignment = RolePermission(
                role_id=role_id,
                permission_id=permission_id
            )
            
            session.add(new_assignment)
            session.commit()
            
            logger.info(f"Assigned permission {permission_id} to role {role_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to assign permission: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_user_roles(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get roles assigned to a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of role dictionaries
        """
        session = self.Session()
        
        try:
            # Get role assignments
            role_assignments = session.query(UserRole, Role).join(
                Role, UserRole.role_id == Role.id
            ).filter(
                UserRole.user_id == user_id
            ).all()
            
            # Format roles
            roles = [
                {
                    "id": role.id,
                    "name": role.name,
                    "description": role.description
                }
                for _, role in role_assignments
            ]
            
            return roles
        except Exception as e:
            logger.error(f"Failed to get user roles: {e}")
            return []
        finally:
            session.close()
    
    def get_role_permissions(self, role_id: int) -> List[Dict[str, Any]]:
        """
        Get permissions assigned to a role.
        
        Args:
            role_id: Role ID
            
        Returns:
            List of permission dictionaries
        """
        session = self.Session()
        
        try:
            # Get permission assignments
            permission_assignments = session.query(RolePermission, Permission).join(
                Permission, RolePermission.permission_id == Permission.id
            ).filter(
                RolePermission.role_id == role_id
            ).all()
            
            # Format permissions
            permissions = [
                {
                    "id": permission.id,
                    "name": permission.name,
                    "resource": permission.resource,
                    "action": permission.action,
                    "description": permission.description
                }
                for _, permission in permission_assignments
            ]
            
            return permissions
        except Exception as e:
            logger.error(f"Failed to get role permissions: {e}")
            return []
        finally:
            session.close()
    
    def check_permission(
        self,
        user_id: int,
        resource: str,
        action: str
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.
        
        Args:
            user_id: User ID
            resource: Resource to access
            action: Action to perform
            
        Returns:
            True if the user has permission, False otherwise
        """
        session = self.Session()
        
        try:
            # Check if the user is an admin
            is_admin = session.query(User.is_admin).filter(User.id == user_id).scalar()
            
            if is_admin:
                return True
            
            # Check for specific permission
            has_permission = session.query(Permission).join(
                RolePermission, Permission.id == RolePermission.permission_id
            ).join(
                UserRole, RolePermission.role_id == UserRole.role_id
            ).filter(
                UserRole.user_id == user_id,
                Permission.resource == resource,
                Permission.action == action
            ).first() is not None
            
            return has_permission
        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
            return False
        finally:
            session.close()

class Encryption:
    """Encryption module for securing sensitive data."""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize the encryption module.
        
        Args:
            key: Encryption key (base64-encoded)
        """
        if key:
            self.key = key.encode()
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt a string.
        
        Args:
            data: String to encrypt
            
        Returns:
            Encrypted data (base64-encoded)
        """
        encrypted_data = self.cipher.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt a string.
        
        Args:
            encrypted_data: Encrypted data (base64-encoded)
            
        Returns:
            Decrypted string
        """
        decrypted_data = self.cipher.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt values in a dictionary.
        
        Args:
            data: Dictionary with values to encrypt
            
        Returns:
            Dictionary with encrypted values
        """
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                encrypted_data[key] = self.encrypt(value)
            elif isinstance(value, dict):
                encrypted_data[key] = self.encrypt_dict(value)
            elif isinstance(value, list):
                encrypted_data[key] = [
                    self.encrypt_dict(item) if isinstance(item, dict)
                    else self.encrypt(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_dict(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt values in a dictionary.
        
        Args:
            encrypted_data: Dictionary with encrypted values
            
        Returns:
            Dictionary with decrypted values
        """
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            if isinstance(value, str):
                try:
                    decrypted_data[key] = self.decrypt(value)
                except Exception:
                    decrypted_data[key] = value
            elif isinstance(value, dict):
                decrypted_data[key] = self.decrypt_dict(value)
            elif isinstance(value, list):
                decrypted_data[key] = [
                    self.decrypt_dict(item) if isinstance(item, dict)
                    else self.decrypt(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                decrypted_data[key] = value
        
        return decrypted_data

class AuditLogger:
    """Audit logging module for tracking actions."""
    
    def __init__(self, db_url: str):
        """
        Initialize the audit logger.
        
        Args:
            db_url: Database URL for audit log storage
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def log_event(
        self,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> Optional[int]:
        """
        Log an event.
        
        Args:
            action: Action performed
            resource: Resource affected
            resource_id: ID of the resource affected
            user_id: ID of the user who performed the action
            username: Username of the user who performed the action
            details: Additional details about the event
            ip_address: IP address of the user
            
        Returns:
            Audit log entry ID or None if logging failed
        """
        session = self.Session()
        
        try:
            # Create audit log entry
            audit_log = AuditLog(
                user_id=user_id,
                username=username,
                action=action,
                resource=resource,
                resource_id=resource_id,
                details=json.dumps(details) if details else None,
                ip_address=ip_address
            )
            
            session.add(audit_log)
            session.commit()
            
            return audit_log.id
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def search_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            user_id: User ID for filtering
            username: Username for filtering
            action: Action for filtering
            resource: Resource for filtering
            resource_id: Resource ID for filtering
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of audit log entries
        """
        session = self.Session()
        
        try:
            # Build query
            query = session.query(AuditLog)
            
            if start_time:
                query = query.filter(AuditLog.timestamp >= start_time)
            
            if end_time:
                query = query.filter(AuditLog.timestamp <= end_time)
            
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            
            if username:
                query = query.filter(AuditLog.username == username)
            
            if action:
                query = query.filter(AuditLog.action == action)
            
            if resource:
                query = query.filter(AuditLog.resource == resource)
            
            if resource_id:
                query = query.filter(AuditLog.resource_id == resource_id)
            
            # Apply pagination
            query = query.order_by(AuditLog.timestamp.desc())
            query = query.limit(limit).offset(offset)
            
            # Format results
            logs = []
            for log in query.all():
                log_entry = {
                    "id": log.id,
                    "user_id": log.user_id,
                    "username": log.username,
                    "action": log.action,
                    "resource": log.resource,
                    "resource_id": log.resource_id,
                    "ip_address": log.ip_address,
                    "timestamp": log.timestamp.isoformat()
                }
                
                if log.details:
                    try:
                        log_entry["details"] = json.loads(log.details)
                    except json.JSONDecodeError:
                        log_entry["details"] = log.details
                
                logs.append(log_entry)
            
            return logs
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            return []
        finally:
            session.close()
    
    def get_log_entry(self, log_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific audit log entry.
        
        Args:
            log_id: ID of the audit log entry
            
        Returns:
            Audit log entry or None if not found
        """
        session = self.Session()
        
        try:
            # Get log entry
            log = session.query(AuditLog).filter(AuditLog.id == log_id).first()
            
            if not log:
                return None
            
            # Format result
            log_entry = {
                "id": log.id,
                "user_id": log.user_id,
                "username": log.username,
                "action": log.action,
                "resource": log.resource,
                "resource_id": log.resource_id,
                "ip_address": log.ip_address,
                "timestamp": log.timestamp.isoformat()
            }
            
            if log.details:
                try:
                    log_entry["details"] = json.loads(log.details)
                except json.JSONDecodeError:
                    log_entry["details"] = log.details
            
            return log_entry
        except Exception as e:
            logger.error(f"Failed to get audit log entry: {e}")
            return None
        finally:
            session.close()

class Security:
    """Central security module that combines authentication, authorization, encryption, and audit logging."""
    
    def __init__(
        self,
        db_url: str,
        jwt_secret: str,
        token_expiry: int = 3600,
        oauth_providers: Optional[Dict[str, Dict[str, str]]] = None,
        encryption_key: Optional[str] = None
    ):
        """
        Initialize the security module.
        
        Args:
            db_url: Database URL for security data storage
            jwt_secret: Secret for JWT token signing
            token_expiry: Token expiry time in seconds
            oauth_providers: Dictionary of OAuth providers and their configurations
            encryption_key: Key for data encryption
        """
        self.auth = Authentication(db_url, jwt_secret, token_expiry, oauth_providers)
        self.authz = Authorization(db_url)
        self.encryption = Encryption(encryption_key)
        self.audit = AuditLogger(db_url)
    
    def setup_default_roles_and_permissions(self) -> None:
        """Set up default roles and permissions."""
        # Create default roles
        admin_role_id = self.authz.create_role(
            name="admin",
            description="Administrator with full access"
        )
        
        data_scientist_role_id = self.authz.create_role(
            name="data_scientist",
            description="Data Scientist with model training and deployment permissions"
        )
        
        data_engineer_role_id = self.authz.create_role(
            name="data_engineer",
            description="Data Engineer with data processing permissions"
        )
        
        viewer_role_id = self.authz.create_role(
            name="viewer",
            description="Viewer with read-only access"
        )
        
        # Create default permissions
        
        # Model permissions
        train_model_permission_id = self.authz.create_permission(
            name="train_model",
            resource="model",
            action="train",
            description="Train a model"
        )
        
        deploy_model_permission_id = self.authz.create_permission(
            name="deploy_model",
            resource="model",
            action="deploy",
            description="Deploy a model to production"
        )
        
        view_model_permission_id = self.authz.create_permission(
            name="view_model",
            resource="model",
            action="view",
            description="View model details"
        )
        
        # Data permissions
        process_data_permission_id = self.authz.create_permission(
            name="process_data",
            resource="data",
            action="process",
            description="Process and transform data"
        )
        
        view_data_permission_id = self.authz.create_permission(
            name="view_data",
            resource="data",
            action="view",
            description="View data"
        )
        
        # Feature permissions
        create_feature_permission_id = self.authz.create_permission(
            name="create_feature",
            resource="feature",
            action="create",
            description="Create a new feature"
        )
        
        view_feature_permission_id = self.authz.create_permission(
            name="view_feature",
            resource="feature",
            action="view",
            description="View feature details"
        )
        
        # Assign permissions to roles
        
        # Admin role has all permissions
        if admin_role_id:
            for permission_id in [
                train_model_permission_id,
                deploy_model_permission_id,
                view_model_permission_id,
                process_data_permission_id,
                view_data_permission_id,
                create_feature_permission_id,
                view_feature_permission_id
            ]:
                if permission_id:
                    self.authz.assign_permission_to_role(admin_role_id, permission_id)
        
        # Data Scientist role
        if data_scientist_role_id:
            for permission_id in [
                train_model_permission_id,
                view_model_permission_id,
                view_data_permission_id,
                create_feature_permission_id,
                view_feature_permission_id
            ]:
                if permission_id:
                    self.authz.assign_permission_to_role(data_scientist_role_id, permission_id)
        
        # Data Engineer role
        if data_engineer_role_id:
            for permission_id in [
                process_data_permission_id,
                view_data_permission_id,
                create_feature_permission_id,
                view_feature_permission_id
            ]:
                if permission_id:
                    self.authz.assign_permission_to_role(data_engineer_role_id, permission_id)
        
        # Viewer role
        if viewer_role_id:
            for permission_id in [
                view_model_permission_id,
                view_data_permission_id,
                view_feature_permission_id
            ]:
                if permission_id:
                    self.authz.assign_permission_to_role(viewer_role_id, permission_id)
    
    def create_initial_admin_user(
        self,
        username: str,
        email: str,
        password: str
    ) -> Optional[int]:
        """
        Create an initial admin user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            
        Returns:
            User ID or None if creation failed
        """
        # Register admin user
        user_id = self.auth.register_user(
            username=username,
            email=email,
            password=password,
            is_admin=True
        )
        
        if not user_id:
            return None
        
        # Assign admin role
        admin_role_id = self.authz.create_role(
            name="admin",
            description="Administrator with full access"
        )
        
        if admin_role_id:
            self.authz.assign_role_to_user(user_id, admin_role_id)
        
        # Log the event
        self.audit.log_event(
            action="create_user",
            resource="user",
            resource_id=str(user_id),
            username="system",
            details={"is_admin": True}
        )
        
        return user_id

# Example usage
if __name__ == "__main__":
    # Initialize security module
    security = Security(
        db_url="sqlite:///security.db",
        jwt_secret="your-jwt-secret-key",
        oauth_providers={
            "google": {
                "client_id": "your-google-client-id",
                "client_secret": "your-google-client-secret",
                "token_url": "https://oauth2.googleapis.com/token",
                "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo"
            }
        }
    )
    
    # Set up default roles and permissions
    security.setup_default_roles_and_permissions()
    
    # Create admin user
    admin_id = security.create_initial_admin_user(
        username="admin",
        email="admin@example.com",
        password="secure-password"
    )
    
    # Register a regular user
    user_id = security.auth.register_user(
        username="user",
        email="user@example.com",
        password="user-password"
    )
    
    # Assign a role to the user
    data_scientist_role_id = security.authz.create_role(
        name="data_scientist",
        description="Data Scientist role"
    )
    
    security.authz.assign_role_to_user(user_id, data_scientist_role_id)
    
    # Authenticate user
    user_info = security.auth.authenticate_user(
        username="user",
        email="user@example.com",
        password="user-password"
    )
    
    if user_info:
        # Generate token
        token = security.auth.generate_token(
            user_id=user_info["id"],
            username=user_info["username"]
        )
        
        # Verify permission
        has_permission = security.authz.check_permission(
            user_id=user_info["id"],
            resource="model",
            action="train"
        )
        
        # Log an event
        security.audit.log_event(
            action="login",
            resource="user",
            resource_id=str(user_info["id"]),
            user_id=user_info["id"],
            username=user_info["username"],
            ip_address="127.0.0.1"
        )
        
        # Encrypt sensitive data
        sensitive_data = {
            "api_key": "secret-api-key",
            "credentials": {
                "username": "service-account",
                "password": "service-password"
            }
        }
        
        encrypted_data = security.encryption.encrypt_dict(sensitive_data)
        decrypted_data = security.encryption.decrypt_dict(encrypted_data)