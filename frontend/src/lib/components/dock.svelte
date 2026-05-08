<script lang="ts">
	import CameraIcon from 'phosphor-svelte/lib/CameraIcon';
	import ClockIcon from 'phosphor-svelte/lib/ClockIcon';
	import GearIcon from 'phosphor-svelte/lib/GearIcon';
</script>

<div class="dock">
	<button>
		<CameraIcon weight="bold" />
		<span class="dock-label">Scan</span>
	</button>

	<button class="dock-active">
		<ClockIcon weight="bold" />
		<span class="dock-label">History</span>
	</button>

	<button>
		<GearIcon />
		<span class="dock-label">Settings</span>
	</button>
</div>
